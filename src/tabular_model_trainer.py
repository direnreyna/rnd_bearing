# src/tabular_model_trainer.py

import logging
import pathlib
import optuna
import mlflow
import joblib
import config
import numpy as np
import pandas as pd
import lightgbm as lgb

from optuna.integration import LightGBMPruningCallback
import lightgbm.callback as lgb_callback
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split ##, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

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
            best_params = self._tune_hyperparameters(X_train, y_train)
            self.logger.info(f"Optuna завершен. Лучшие параметры: {best_params}")
            ### best_params = self._tune_hyperparameters(X_train, y_train, config.LGBM_TUNING_PARAMS)

        # 3. Обучение модели
        self.model = self._train_model(X_train, y_train, best_params)

        ### # 4. Сохранение финальной модели
        ### self._save_model()
        
        ### # 5. Оценка
        ### y_pred, mse = self._evaluate_model(X_test, y_test)
        ### self.logger.info(f"Оценка LightGBM на тестовом наборе (MSE): {mse:.4f}")

        # 6. Извлечение важности признаков
        feature_importance = pd.Series(self.model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        
        # 7. Применение Feature Selection (если включено)
        if config.ENABLE_FEATURE_SELECTION:
            top_features_list = feature_importance.head(config.N_TOP_FEATURES).index.tolist()
            X_train = X_train[top_features_list]
            X_test = X_test[top_features_list]
            self.logger.info(f"Применен Feature Selection. Использовано TOP-{config.N_TOP_FEATURES} признаков. ПЕРЕОБУЧЕНИЕ...")
            
            # Переобучаем модель на меньшем наборе признаков
            self.model = self._train_model(X_train, y_train, best_params)
            # Пересчитываем важность (хотя она будет та же)
            feature_importance = pd.Series(self.model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        
        # 8. Сохранение финальной модели
        self._save_model()

        # 9. Оценка
        y_pred, rmse = self._evaluate_model(X_test, y_test)
        self.logger.info(f"Оценка LightGBM на тестовом наборе (RMSE): {rmse:.4f} часов")

        return self.model, X_test, y_test, y_pred, feature_importance
    
###    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, Any]) -> Dict[str, Any]: ## ДОБАВЛЕН БЛОК
###        """
###        Использует RandomizedSearchCV для поиска оптимальных гиперпараметров.
###        """
###        self.logger.info(f"Запуск RandomizedSearchCV: {config.N_ITER_SEARCH} итераций...")
###
###        # Базовая модель для поиска
###        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
###
###        # Randomized Search
###        random_search = RandomizedSearchCV(
###            estimator=lgbm,
###            param_distributions=param_grid,
###            n_iter=config.N_ITER_SEARCH,
###            scoring='neg_mean_squared_error',
###            cv=3, # 3-кратная кросс-валидация
###            verbose=1,
###            random_state=42,
###            n_jobs=-1
###        )
###        
###        random_search.fit(X_train, y_train)
###
###        self.logger.info(f"Тюнинг завершен. Лучший MSE: {-random_search.best_score_:.4f}")
###        self.logger.info(f"Лучшие параметры: {random_search.best_params_}")
###
###        return random_search.best_params_

    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Использует Optuna для Байесовской оптимизации гиперпараметров и логирует результаты в MLflow.
        """
        self.logger.info(f"Запуск Optuna Байесовской оптимизации: {config.OPTUNA_N_TRIALS} проб.")

        # Определение objective-функции
        def objective(trial: optuna.Trial):
            with mlflow.start_run(nested=True) as run:
                # 1. Определение пространства поиска (на основе config.LGBM_OPTUNA_PARAMS)
                param = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'n_estimators': trial.suggest_int('n_estimators', config.LGBM_OPTUNA_PARAMS['n_estimators'][0], config.LGBM_OPTUNA_PARAMS['n_estimators'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', config.LGBM_OPTUNA_PARAMS['learning_rate'][0], config.LGBM_OPTUNA_PARAMS['learning_rate'][1]),
                    'num_leaves': trial.suggest_int('num_leaves', config.LGBM_OPTUNA_PARAMS['num_leaves'][0], config.LGBM_OPTUNA_PARAMS['num_leaves'][1]),
                    'max_depth': trial.suggest_int('max_depth', config.LGBM_OPTUNA_PARAMS['max_depth'][0], config.LGBM_OPTUNA_PARAMS['max_depth'][1]),
                    'min_child_samples': trial.suggest_int('min_child_samples', config.LGBM_OPTUNA_PARAMS['min_child_samples'][0], config.LGBM_OPTUNA_PARAMS['min_child_samples'][1]),
                    'reg_alpha': trial.suggest_float('reg_alpha', config.LGBM_OPTUNA_PARAMS['reg_alpha'][0], config.LGBM_OPTUNA_PARAMS['reg_alpha'][1]),
                    'reg_lambda': trial.suggest_float('reg_lambda', config.LGBM_OPTUNA_PARAMS['reg_lambda'][0], config.LGBM_OPTUNA_PARAMS['reg_lambda'][1]),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1,
                }
                
                # Логирование параметров в MLflow
                mlflow.log_params(param)

                # 2. Обучение с кросс-валидацией
                lgbm = lgb.LGBMRegressor(**param)
                
                # Используем кросс-валидацию (KFold, 3-Fold)
                # KFold не подходит из-за временных рядов, но для Optuna KFold чаще используется.
                # Для упрощения MVP, пока используем train_test_split внутри Optuna
                
                # Разбиваем train set на Optuna_train и Optuna_val (20% на валидацию)
                X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
                )

                ### lgbm.fit(X_opt_train, y_opt_train)
                lgbm.fit(X_opt_train, y_opt_train, eval_set=[(X_opt_val, y_opt_val)], eval_metric='rmse', callbacks=[LightGBMPruningCallback(trial, 'rmse'), lgb_callback.early_stopping(100)])

                # 3. Предсказание и оценка на валидационном наборе
                y_pred = lgbm.predict(X_opt_val)
                rmse = np.sqrt(mean_squared_error(y_opt_val, np.array(y_pred)))
                
                # Логирование метрики в MLflow
                mlflow.log_metric("rmse", rmse)
                
                return rmse # Optuna минимизирует возвращаемое значение

        # Запуск Optuna Study
        study = optuna.create_study(
            direction='minimize',
            study_name=f'lgbm_rul_study_{self.experiment_name}',
            storage=config.OPTUNA_STORAGE_URI,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT, show_progress_bar=True)

        self.logger.info(f"Тюнинг завершен. Лучший RMSE: {study.best_value:.4f}")
        self.logger.info(f"Лучшие параметры: {study.best_params}")

        return study.best_params

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

        # Объединяем все колонки для исключения
        all_cols_to_exclude = cols_to_drop + freq_cols

        # 1.2: Создание списка колонок, которые МЫ ХОТИМ ИСПОЛЬЗОВАТЬ
        feature_candidates = [col for col in df.columns if col not in all_cols_to_exclude]
        
        final_feature_cols = []
        
        # Логика Feature Ablation
        if config.USE_D0_FEATURES:
            # Centered Amplitudes: (d0) - ищем '_centered'
            final_feature_cols.extend([col for col in feature_candidates if '_centered' in col])
        
        if config.USE_D1_FEATURES:
            # Velocities: (d1) - ищем '_velo'
            final_feature_cols.extend([col for col in feature_candidates if '_velo' in col])

        if config.USE_D2_FEATURES:
            # Accelerations: (d2) - ищем '_accel'
            final_feature_cols.extend([col for col in feature_candidates if '_accel' in col])
        
        X = df[final_feature_cols]
        
        self.logger.info(f"Количество признаков для обучения: {X.shape[1]} (d0: {config.USE_D0_FEATURES}, d1: {config.USE_D1_FEATURES}, d2: {config.USE_D2_FEATURES})")
        
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
        assert self.model is not None ## Для устранения Pylance Optional warning
        y_pred = self.model.predict(X_test)
        # Рассчитываем MSE и RMSE
        mse = mean_squared_error(y_test, np.array(y_pred))
        rmse = np.sqrt(mse)

        # Рассчитываем R2 для справки
        r2 = r2_score(y_test, np.array(y_pred))

        self.logger.info(f"R2 (Коэффициент детерминации): {r2:.4f}")

        # Преобразуем numpy-массив в pandas Series для удобства дальнейшей работы
        ### y_pred_series = pd.Series(y_pred, index=y_test.index)

        y_pred_series = pd.Series(np.array(y_pred), index=y_test.index)

        return y_pred_series, rmse