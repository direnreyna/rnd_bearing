# src/tabular_model_trainer.py


import logging
import pathlib
import pandas as pd
import lightgbm as lgb

from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib

import config

class TabularModelTrainer:
    """
    Класс для подготовки данных, обучения и оценки базовой модели (LightGBM)
    для задачи регрессии RUL.
    """

    def __init__(self, logger: logging.Logger, experiment_name: str):
        """
        Инициализирует трейнер.

        Args:
            logger (logging.Logger): Экземпляр логгера.
            experiment_name (str): Имя текущего эксперимента.
        """
        self.logger = logger
        self.experiment_name = experiment_name
        self.model: Optional[lgb.LGBMRegressor] = None

    def run(self, df: pd.DataFrame) -> Tuple[lgb.LGBMRegressor, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Главный метод, управляющий подготовкой данных, обучением и оценкой.

        Args:
            df (pd.DataFrame): Обогащенный DataFrame со спектральными признаками и RUL.

        Returns:
            Tuple: (Обученная модель, X_test, y_test, y_pred, feature_importance)
        """
        self.logger.info("Запуск обучения базовой модели (LightGBM) для предсказания RUL.")

        # 1. Подготовка данных
        X_train, X_test, y_train, y_test = self._prepare_data(df)


        # 2. Тюнинг гиперпараметров (если включено)
        best_params = {}
        if config.ENABLE_MODEL_TUNING:
            best_params = self._tune_hyperparameters(X_train, y_train, config.LGBM_TUNING_PARAMS)

        # 3. Обучение модели
        self.model = self._train_model(X_train, y_train, best_params)

        # 4. Сохранение финальной модели
        self._save_model()

        # 5. Оценка
        y_pred, mse = self._evaluate_model(X_test, y_test)
        self.logger.info(f"Оценка LightGBM на тестовом наборе (MSE): {mse:.4f}")

        # 6. Извлечение важности признаков
        feature_importance = pd.Series(self.model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        
        return self.model, X_test, y_test, y_pred, feature_importance
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, Any]) -> Dict[str, Any]: ## ДОБАВЛЕН БЛОК
        """
        Использует RandomizedSearchCV для поиска оптимальных гиперпараметров.
        """
        self.logger.info(f"Запуск RandomizedSearchCV: {config.N_ITER_SEARCH} итераций...")

        # Базовая модель для поиска
        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

        # Randomized Search
        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=config.N_ITER_SEARCH,
            scoring='neg_mean_squared_error',
            cv=3, # 3-кратная кросс-валидация
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)

        self.logger.info(f"Тюнинг завершен. Лучший MSE: {-random_search.best_score_:.4f}")
        self.logger.info(f"Лучшие параметры: {random_search.best_params_}")

        return random_search.best_params_

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Выделяет признаки, целевую переменную и разбивает датасет.
        """
        self.logger.info("Подготовка данных для LightGBM: выделение X, y и train/test split.")
        
        # 1. Определение признаков (X) и целевой переменной (y)
        y = df['RUL_hours']
        
        # Удаляем мета-колонки, колонку RUL, и колонки с частотами (оставляем только амплитуды и их производные)
        cols_to_drop = [
            'RUL_hours', 'timestamp', 'bearing', 'window_id'
        ]
        # Дополнительно удаляем колонки с частотами, т.к. они сложнее для интерпретации LightGBM
        freq_cols = [col for col in df.columns if '_freq' in col]
        
        X = df.drop(columns=cols_to_drop + freq_cols, errors='ignore')
        
        self.logger.info(f"Количество признаков для обучения: {X.shape[1]}")
        
        # 2. Разделение на обучающую и тестовую выборки
        # Используем train_test_split. Обычно для RUL лучше делать это по времени,
        # но для Baseline Model подойдет и случайное разделение (для простоты).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True
        )

        # 3. Убеждаемся, что y_train и y_test - это Series, а не DataFrame
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        self.logger.info(f"Размер обучающей выборки: {len(X_train)} строк")
        self.logger.info(f"Размер тестовой выборки: {len(X_test)} строк")
        
        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, custom_params: Dict[str, Any] = None) -> lgb.LGBMRegressor:
        """
        Инициализирует и обучает модель LightGBM.
        """
        self.logger.info("Обучение LightGBMRegressor...")
        
        # Базовые (Baseline) гиперпараметры LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse', # Root Mean Squared Error (Квадратный корень из MSE)
            'n_estimators': 500,
            'learning_rate': 0.05,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1 # Отключаем логирование LightGBM
        }

        # Обновляем параметры, если были найдены лучшие в ходе тюнинга
        if custom_params:
            params.update(custom_params)
            self.logger.info(f"Обучение с лучшими параметрами: {custom_params}")

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        self.logger.info("Обучение модели LightGBM завершено.")
        return model

    def _save_model(self):
        """
        Сохраняет обученную модель на диск с помощью joblib.
        """
        save_path = config.MODEL_FILEPATH
        joblib.dump(self.model, save_path)
        self.logger.info(f"Обученная модель LightGBM сохранена: {save_path}")
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.Series, float]:
        """
        Делает предсказания и оценивает модель.
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Преобразуем numpy-массив в pandas Series для удобства дальнейшей работы
        y_pred_series = pd.Series(y_pred, index=y_test.index)

        return y_pred_series, mse