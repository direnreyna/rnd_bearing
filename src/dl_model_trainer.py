# src/dl_model_trainer.py

import logging
import pathlib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import mlflow

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from .dl_model_architect import DLModelArchitect
import config
import optuna

class DLModelTrainer:
    """
    Класс для обучения и тюнинга DL-моделей (LSTM/CNN).
    """

    def __init__(self, logger: logging.Logger, experiment_name: str, sequence_length: int, lstm_units: int, epochs: int, batch_size: int):
        self.logger = logger
        self.experiment_name = experiment_name
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.architect = DLModelArchitect(logger)
        self.best_model: Model = None

    def run(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Model:
        """
        Главный метод: запускает Optuna тюнинг и финальное обучение.
        """
        self.logger.info("Запуск DL обучения и тюнинга (Keras/TensorFlow).")
        
        # 1. Запуск Optuna Tuning
        best_params = self._tune_hyperparameters(X_train, y_train, X_val, y_val)
        
        # 2. Финальное обучение лучшей модели
        self.best_model = self._train_model(X_train, y_train, X_val, y_val, best_params)
        
        # 3. Сохранение модели и весов
        self._save_model(self.best_model)
        
        return self.best_model

    def finetune(self, mode: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, X_test_meta: pd.DataFrame) -> Model:
        """
        Метод для дообучения (finetune) DL-модели.
        Загружает существующую модель/веса, затем дообучает.
        """
        self.logger.info("Запуск режима дообучения (finetune) DL-модели.")

        # 1. Загрузка существующей модели или весов
        # Для загрузки архитектуры нам нужны X_shape для определения num_features
        # X_train, X_val, X_test уже 3D (samples, sequence_length, num_features)
        loaded_model = self._load_model_or_weights(mode, X_train.shape)

        self.best_model = loaded_model # Устанавливаем загруженную модель как best_model

        # 2. Компиляция модели для дообучения (используем DL_FINETUNE_LEARNING_RATE)
        from tensorflow.keras.optimizers import Adam
        # Используем learning_rate для дообучения из конфига
        optimizer = Adam(learning_rate=config.DL_LEARNING_RATE)
        self.best_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.logger.info(f"Модель скомпилирована для дообучения с learning_rate={config.DL_LEARNING_RATE}")

        # 3. Подготовка данных для обучения на полном наборе (Train+Val)
        ### X_train_full = np.concatenate((X_train, X_val), axis=0)
        ### y_train_full = np.concatenate((y_train, y_val), axis=0)

        data_to_concatenate_X = [X_train]
        if X_val.size > 0: # Только если X_val не пуст
            data_to_concatenate_X.append(X_val)
        X_train_full = np.concatenate(data_to_concatenate_X, axis=0)

        data_to_concatenate_y = [y_train]
        if y_val.size > 0: # Только если y_val не пуст
            data_to_concatenate_y.append(y_val)
        y_train_full = np.concatenate(data_to_concatenate_y, axis=0)

        # 4. Callbacks и обучение
        early_stop = EarlyStopping(monitor='loss', patience=config.DL_EARLY_STOPPING_PATIENCE, verbose=1, mode='min')
        
        self.logger.info("Запуск дообучения...")
        self.best_model.fit(
            X_train_full, y_train_full,
            epochs=self.epochs, # Используем максимальное количество эпох из конфига
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        self.logger.info("Дообучение завершено.")

        # 5. Сохранение дообученной модели
        self._save_model(self.best_model)
        
        return self.best_model

    def run_inference_mode(self, mode: str, X_test_3D: np.ndarray, X_shape_for_arch: Tuple[int, int]) -> np.ndarray:
        """
        Метод для выполнения инференса (предсказаний) на DL-модели.
        Загружает существующую модель/веса, затем делает предсказания.
        """
        self.logger.info("Запуск режима инференса (predict) DL-модели.")

        # 1. Загрузка существующей модели или весов
        # Для загрузки архитектуры нам нужны X_shape для определения num_features
        # X_test_3D уже 3D (samples, sequence_length, num_features)
        loaded_model = self._load_model_or_weights(mode, X_shape_for_arch)
        
        self.best_model = loaded_model # Устанавливаем загруженную модель как best_model для predict

        if self.best_model is None:
            self.logger.error("Не удалось загрузить модель для инференса. Проверьте пути и наличие файлов.")
            raise RuntimeError("Модель DL не инициализирована для инференса.")
        
        # 2. Выполнение предсказаний
        y_pred = self.predict(X_test_3D)
        self.logger.info("Предсказания для инференса выполнены.")
        
        return y_pred

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Тюнинг гиперпараметров DL-архитектуры через Optuna.
        """
        self.logger.info(f"Запуск Optuna DL-оптимизации: {config.OPTUNA_N_TRIALS} проб.")
        
        num_features = X_train.shape[2]
        
        def objective(trial: optuna.Trial):
            with mlflow.start_run(nested=True) as run:
                # 1. Предложение гиперпараметров
                lstm_units = trial.suggest_int('lstm_units', 32, 128)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
                
                # 2. Сборка модели
                model = self.architect.build_cnn_lstm_attention(
                    sequence_length=self.sequence_length,
                    num_features=num_features,
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                
                ### # Применение предложенного learning_rate к оптимизатору
                ### from tensorflow.keras.optimizers import Adam
                ### optimizer = Adam(learning_rate=learning_rate)
                ### model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                
                # 3. Callbacks и обучение
                early_stop = EarlyStopping(monitor='val_loss', patience=config.DL_EARLY_STOPPING_PATIENCE, verbose=0, mode='min')
                
                history = model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=0 # Отключаем вывод Keras
                )

                # 4. Получение метрики и логирование
                best_val_loss = min(history.history['val_loss'])
                best_val_rmse = np.sqrt(best_val_loss) # RMSE = sqrt(MSE)
                
                mlflow.log_params(trial.params)
                mlflow.log_metric("rmse", best_val_rmse)
                mlflow.log_metric("val_loss", best_val_loss)
                
                return best_val_rmse # Optuna минимизирует RMSE

        # Запуск Optuna Study
        study = optuna.create_study(
            direction='minimize',
            study_name=f'dl_rul_study_{self.experiment_name}',
            storage=config.OPTUNA_STORAGE_URI,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT, show_progress_bar=True)
        
        self.logger.info(f"DL Тюнинг завершен. Лучший RMSE: {study.best_value:.4f}")
        self.logger.info(f"Лучшие параметры: {study.best_params}")
        
        return study.best_params

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, best_params: Dict[str, Any]) -> Model:
        """
        Финальное обучение лучшей модели на полном наборе Train+Val.
        """
        self.logger.info("DL Финальное обучение...")
        
        num_features = X_train.shape[2]
        
        # 1. Сборка лучшей модели
        model = self.architect.build_cnn_lstm_attention(
            sequence_length=self.sequence_length,
            num_features=num_features,
            lstm_units=best_params.get('lstm_units', self.lstm_units),
            dropout_rate=best_params.get('dropout_rate', 0.2),
            learning_rate=best_params.get('learning_rate', config.DL_LEARNING_RATE)
        )

        ### # 2. Компиляция с лучшим LR
        ### from tensorflow.keras.optimizers import Adam
        ### optimizer = Adam(learning_rate=best_params.get('learning_rate', config.DL_LEARNING_RATE))
        ### model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # 3. Callbacks и обучение на Train+Val
        X_train_full = np.concatenate((X_train, X_val), axis=0)
        y_train_full = np.concatenate((y_train, y_val), axis=0)
        
        early_stop = EarlyStopping(monitor='loss', patience=config.DL_EARLY_STOPPING_PATIENCE, verbose=1, mode='min')

        model.fit(
            X_train_full, y_train_full,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.logger.info("DL Финальное обучение завершено.")
        return model

    def _get_model_paths(self) -> Dict[str, pathlib.Path]:
        """
        Формирует пути для сохранения модели и весов на основе конфигурации.
        """
        model_name_prefix = f"{config.MODEL_TYPE}_{config.EXPERIMENT_NAME}"
        
        model_keras_filepath = config.DL_MODEL_DIR / f"{model_name_prefix}_model.keras"
        model_h5_filepath = config.DL_MODEL_DIR / f"{model_name_prefix}_model.h5" # Для обратной совместимости
        weights_h5_filepath = config.DL_MODEL_DIR / f"{model_name_prefix}.weights.h5"

        return {
            'model_keras': model_keras_filepath,
            'model_h5': model_h5_filepath,
            'weights_h5': weights_h5_filepath,
        }

    def _save_model(self, model: Model):
        """
        Сохраняет обученную модель в новом формате .keras и веса в .h5.
        """
        paths = self._get_model_paths()
        
        # Сохраняем полную модель в формате .keras
        model.save(paths['model_keras'])
        self.logger.info(f"DL Модель сохранена в формате .keras: {paths['model_keras']}")
        
        # Сохраняем только веса в формате .h5 (для дообучения)
        model.save_weights(paths['weights_h5'])
        self.logger.info(f"DL Веса модели сохранены в формате .h5: {paths['weights_h5']}")

        # Дополнительно, для обратной совместимости, можно сохранить модель в .h5,
        # но TensorFlow/Keras уже предупреждает о Legacy-формате.
        # Пока оставляем возможность, если это нужно для специфических целей.
        # model.save(paths['model_h5'])
        # self.logger.info(f"DL Модель сохранена в формате .h5 (Legacy): {paths['model_h5']}")

    def _load_model_or_weights(self, mode: str, X_shape_for_arch: Tuple[int, int]) -> Model:
        """
        Загружает модель или веса в зависимости от режима и наличия файлов.
        """
        paths = self._get_model_paths()
        model_keras_filepath = paths['model_keras']
        model_h5_filepath = paths['model_h5']
        weights_h5_filepath = paths['weights_h5']

        num_features = X_shape_for_arch[2] if len(X_shape_for_arch) == 3 else X_shape_for_arch[1] # Для архитектуры нужен (sequence_length, num_features)

        self.logger.info(f"Попытка загрузки модели/весов для режима '{mode}'.")

        # 1. Попытка загрузить полную модель (формат .keras или .h5)
        loaded_model = None
        if model_keras_filepath.exists():
            self.logger.info(f"Загружаю полную модель из: {model_keras_filepath} (формат .keras)")
            loaded_model = load_model(model_keras_filepath)
        elif model_h5_filepath.exists():
            self.logger.warning(f"Загружаю полную модель из: {model_h5_filepath} (устаревший формат .h5)")
            loaded_model = load_model(model_h5_filepath)

        if loaded_model:
            self.logger.info("Полная модель успешно загружена.")

        # 2. Если полная модель не найдена, строим архитектуру и ищем веса
        self.logger.info("Полная модель не найдена. Создаю новую архитектуру модели.")
        new_model = self.architect.build_cnn_lstm_attention(
            sequence_length=self.sequence_length,
            num_features=num_features,
            lstm_units=self.lstm_units, # Используем дефолтные или конфигурированные LSTM units
            dropout_rate=0.2,
            learning_rate=config.DL_LEARNING_RATE
        )

        if weights_h5_filepath.exists():
            self.logger.info(f"Загружаю веса в новую архитектуру из: {weights_h5_filepath}")
            try:
                new_model.load_weights(weights_h5_filepath)
                self.logger.info("Веса успешно загружены в новую архитектуру.")

                # Если веса загружены, модель нужно скомпилировать
                from tensorflow.keras.optimizers import Adam
                optimizer = Adam(learning_rate=config.DL_LEARNING_RATE) # Используем базовый LR для компиляции
                new_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                self.logger.info("Модель скомпилирована после загрузки весов.")

            except ValueError as e:
                self.logger.error(f"Ошибка при загрузке весов: {e}. Модель будет обучаться с нуля.")
        else:
            self.logger.warning("Файлы модели и весов не найдены. Модель будет обучаться с нуля.")

        return new_model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Делает предсказание RUL на DL-модели.
        """
        if self.best_model is None:
###             model_filepath = config.PROCESSED_DATA_DIR / f'{self.experiment_name}_dl_model.h5'
###             self.best_model = load_model(model_filepath)
###             self.logger.info(f"DL Модель загружена для предсказания: {model_filepath}")

            self.logger.error("Модель не обучена или не загружена. Невозможно сделать предсказание.")
            raise RuntimeError("Модель DL не инициализирована для предсказания.")
        
        # Предсказание и сведение 3D-выхода к 2D (если Attention Layer возвращает последовательности)
        y_pred_3d = self.best_model.predict(X_test, verbose=0)
        
        # Сведение (pooling) - для нашего Attention-слоя нам нужен последний элемент последовательности
        # Keras 2D-выход: (Samples, Sequence Length, 1) -> берем последний элемент Sequence Length
        # Keras Output (если без Global Pooling) -> (Samples, Time Steps, 1)
        
        # Для нашей архитектуры (Dense после Attention) выход уже (Samples, 1) или (Samples, Sequence Length, 1)
        
        # Если выход 3D (Samples, T, 1) - берем последний T (если Attention не Pooling)
        if y_pred_3d.ndim == 3:
            y_pred = y_pred_3d[:, -1, 0]
        else:
            y_pred = y_pred_3d.flatten()

        return y_pred