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
from catboost import CatBoostRegressor

from optuna.integration import LightGBMPruningCallback
import lightgbm.callback as lgb_callback
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split ##, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def asymmetric_mse_loss(y_pred, y_true, weight=None, group=None):
    """
    Кастомная функция потерь: Асимметричный MSE.
    Штрафует переоценку (RUL_Pred > RUL_True) в 10 раз сильнее.
    """
    alpha = 10.0
    residual = y_pred - y_true
    
    # Scale = alpha (10.0) для остатков > 0 (Overestimation - ОПАСНО)
    scale = np.where(residual > 0, alpha, 1.0) 
    
    # LightGBM ожидает градиент (первая производная) и гессиан (вторая производная)
    grad = scale * 2.0 * residual
    hess = scale * 2.0
    
    return grad, hess

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
        X_train, X_test_meta, y_train, y_test = self._prepare_data(df)
        
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
        
        X_test_model = X_test_meta.drop(columns=['timestamp', 'bearing'])

        # 7. Применение Feature Selection (если включено)
        if config.ENABLE_FEATURE_SELECTION:
            top_features_list = feature_importance.head(config.N_TOP_FEATURES).index.tolist()
            X_train = X_train[top_features_list]
            X_test_model = X_test_model[top_features_list]
            self.logger.info(f"Применен Feature Selection. Использовано TOP-{config.N_TOP_FEATURES} признаков. ПЕРЕОБУЧЕНИЕ...")
            
            # Переобучаем модель на меньшем наборе признаков
            self.model = self._train_model(X_train, y_train, best_params)
            # Пересчитываем важность (хотя она будет та же)
            feature_importance = pd.Series(self.model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        
        # 8. Сохранение финальной модели
        self._save_model()

        # 9. Оценка
        y_pred, rmse = self._evaluate_model(X_test_model, y_test)
        self.logger.info(f"Оценка LightGBM на тестовом наборе (RMSE): {rmse:.4f} часов")

        return self.model, X_test_meta, y_test, y_pred, feature_importance
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Использует Optuna для Байесовской оптимизации гиперпараметров и логирует результаты в MLflow.
        """
        self.logger.info(f"Запуск Optuna Байесовской оптимизации: {config.OPTUNA_N_TRIALS} проб.")

        # Определение objective-функции
        def objective(trial: optuna.Trial):
            with mlflow.start_run(nested=True) as run:
                # Разбиваем train set на Optuna_train и Optuna_val (20% на валидацию)
                X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

                # 1. Определение пространства поиска (на основе config.LGBM_OPTUNA_PARAMS)
                if config.MODEL_TYPE == 'LGBM':
                    model_params = config.LGBM_OPTUNA_PARAMS
                    objective_type = 'regression'
                    metric_type = 'rmse'
                    
                    param = {
                        'objective': objective_type,
                        'metric': metric_type,
                        'n_estimators': trial.suggest_int('n_estimators', model_params['n_estimators'][0], model_params['n_estimators'][1]),
                        'learning_rate': trial.suggest_float('learning_rate', model_params['learning_rate'][0], model_params['learning_rate'][1]),
                        'num_leaves': trial.suggest_int('num_leaves', model_params['num_leaves'][0], model_params['num_leaves'][1]),
                        'max_depth': trial.suggest_int('max_depth', model_params['max_depth'][0], model_params['max_depth'][1]),
                        'min_child_samples': trial.suggest_int('min_child_samples', model_params['min_child_samples'][0], model_params['min_child_samples'][1]),
                        'reg_alpha': trial.suggest_float('reg_alpha', model_params['reg_alpha'][0], model_params['reg_alpha'][1]),
                        'reg_lambda': trial.suggest_float('reg_lambda', model_params['reg_lambda'][0], model_params['reg_lambda'][1]),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1,
                    }
                    model_instance = lgb.LGBMRegressor(**param)
                    
                    # Callbacks для LGBM
                    callbacks_list = [LightGBMPruningCallback(trial, metric_type), lgb_callback.early_stopping(100)]
                    fit_kwargs = {'eval_set': [(X_opt_val, y_opt_val)], 'eval_metric': metric_type, 'callbacks': callbacks_list}
                    
                elif config.MODEL_TYPE == 'CATB':
                    model_params = config.CATB_TUNING_PARAMS
                    metric_type = 'RMSE'
                    
                    param = {
                        'objective': 'RMSE',
                        'eval_metric': metric_type,
                        'iterations': trial.suggest_int('iterations', model_params['iterations'][0], model_params['iterations'][1]),
                        'learning_rate': trial.suggest_loguniform('learning_rate', model_params['learning_rate'][0], model_params['learning_rate'][1]),
                        'depth': trial.suggest_int('depth', model_params['depth'][0], model_params['depth'][1]),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', model_params['l2_leaf_reg'][0], model_params['l2_leaf_reg'][1], log=True),
                        'border_count': trial.suggest_int('border_count', model_params['border_count'][0], model_params['border_count'][1]),
                        'random_seed': 42,
                        'thread_count': -1,
                        'verbose': 0,
                    }
                    model_instance = CatBoostRegressor(**param)
                    
                    # Callbacks для CatBoost
                    fit_kwargs = {'eval_set': [(X_opt_val, y_opt_val)], 'early_stopping_rounds': 100, 'verbose': 0}
                
                else:
                    raise ValueError(f"Неизвестный тип модели: {config.MODEL_TYPE}")

                # Логирование параметров в MLflow
                mlflow.log_params(param)
                
                # Используем кросс-валидацию (KFold, 3-Fold)
                # KFold не подходит из-за временных рядов, но для Optuna KFold чаще используется.
                # Для упрощения MVP, пока используем train_test_split внутри Optuna
                model_instance.fit(X_opt_train, y_opt_train, **fit_kwargs)

                # 3. Предсказание и оценка на валидационном наборе
                y_pred = model_instance.predict(X_opt_val)
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
            'RUL_hours', 'window_id'
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
            final_feature_cols.extend([col for col in feature_candidates if '_centered' in col and 'ch1_' in col])
        
        if config.USE_D1_FEATURES:
            # Velocities: (d1) - ищем '_velo'
            final_feature_cols.extend([col for col in feature_candidates if '_velo' in col and 'ch1_' in col])

        if config.USE_D2_FEATURES:
            # Accelerations: (d2) - ищем '_accel'
            final_feature_cols.extend([col for col in feature_candidates if '_accel' in col and 'ch1_' in col])

        final_cols_for_split = final_feature_cols + ['timestamp', 'bearing']
        X = df[final_cols_for_split]
        
        self.logger.info(f"Количество признаков для обучения: {X.shape[1]} (d0: {config.USE_D0_FEATURES}, d1: {config.USE_D1_FEATURES}, d2: {config.USE_D2_FEATURES})")
        
        # 2. Разделение на обучающую и тестовую выборки
        # Используем train_test_split. Обычно для RUL лучше делать это по времени,
        # но для Baseline Model подойдет и случайное разделение (для простоты).
        ### X_train, X_test, y_train, y_test = train_test_split(
        ###     X, y, test_size=0.3, random_state=42, shuffle=True
        ### )
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True
        )

        # 3. Выделение метаданных из X_test_full и удаление их из X_train_full/X_test_full
        meta_cols = ['timestamp', 'bearing']

        X_test_meta = X_test_full.copy()
        X_train = X_train_full.drop(columns=meta_cols)
        X_test = X_test_full.drop(columns=meta_cols)

        # 4. Убеждаемся, что y_train и y_test - это Series, а не DataFrame
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        self.logger.info(f"Размер обучающей выборки: {len(X_train)} строк")
        self.logger.info(f"Размер тестовой выборки: {len(X_test)} строк")

        return X_train, X_test_meta, y_train, y_test

    def _prepare_predict_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]: ## ДОБАВЛЕНО БЛОК
        """
        Подготавливает данные для предсказания (без train/test split).
        Возвращает X (с метаданными) и y (RUL).
        """
        self.logger.info("Подготовка данных для предсказания: выделение X, y (без split).")
        
        # 1. Определение фич для обучения
        all_cols_to_exclude = [
            'RUL_hours', 'window_id'
        ]
        freq_cols = [col for col in df.columns if '_freq' in col]
        meta_cols = ['timestamp', 'bearing']

        cols_to_exclude = all_cols_to_exclude + freq_cols
        
        # Создание списка фич
        feature_candidates = [col for col in df.columns if col not in cols_to_exclude]
        final_feature_cols = []
        
        if config.USE_D0_FEATURES:
            final_feature_cols.extend([col for col in feature_candidates if '_centered' in col and 'ch1_' in col])
        if config.USE_D1_FEATURES:
            final_feature_cols.extend([col for col in feature_candidates if '_velo' in col and 'ch1_' in col])
        if config.USE_D2_FEATURES:
            final_feature_cols.extend([col for col in feature_candidates if '_accel' in col and 'ch1_' in col])

        # 2. Выделение X с фичами, RUL и метаданными
        X_with_meta = df[final_feature_cols + meta_cols + ['RUL_hours']].copy()

        # 3. Удаляем строки с NaN (здоровые подшипники, которые не достигли TOA)
        X_with_meta = X_with_meta.dropna(subset=['RUL_hours'])
        
        # 4. Финальные X и y
        y = X_with_meta['RUL_hours'].squeeze()
        X = X_with_meta.drop(columns=['RUL_hours'])

        self.logger.info(f"Размер датасета для предсказания: {len(X)} строк")
        
        return X, y

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, custom_params: Dict[str, Any] = None) -> lgb.LGBMRegressor:
        """
        Инициализирует и обучает модель LightGBM.
        """
        self.logger.info(f"Обучение {config.MODEL_TYPE} Regressor...")
        
        # Базовые (Baseline) гиперпараметры LightGBM
        if config.MODEL_TYPE == 'LGBM':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            model_class = lgb.LGBMRegressor
        elif config.MODEL_TYPE == 'CATB':
            params = {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'random_seed': 42,
                'thread_count': -1,
                'verbose': 0
            }
            model_class = CatBoostRegressor
        else:
            raise ValueError(f"Неизвестный тип модели: {config.MODEL_TYPE}")

###         params = {
###             'objective': 'regression',
###             'metric': 'rmse', # Root Mean Squared Error (Квадратный корень из MSE)
###             'n_estimators': 500,
###             'learning_rate': 0.05,
###             'random_state': 42,
###             'n_jobs': -1,
###             'verbose': -1 # Отключаем логирование LightGBM
###         }

        # Обновляем параметры, если были найдены лучшие в ходе тюнинга
        if custom_params:
            params.update(custom_params)
            self.logger.info(f"Обучение с лучшими параметрами: {custom_params}")

        model = model_class(**params)
        model.fit(X_train, y_train)
        
        self.logger.info(f"Обучение модели {config.MODEL_TYPE} завершено.")
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