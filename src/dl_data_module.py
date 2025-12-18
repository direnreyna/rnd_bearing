# src/dl_data_module.py

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from .dl_data_preprocessor import DLDataPreprocessor
import config

class DLDataModule:
    """
    Оркестратор DL-пайплайна данных.
    Выполняет: 
    1. Train/Test/Validation Split (поскольку Sequencing меняет размерность).
    2. Вызывает DLDataPreprocessor для Normalization и Sequencing.
    """

    def __init__(self, logger: logging.Logger, sequence_length: int, test_size: float = 0.3, val_size: float = 0.2, config=None):
        """
        Инициализирует Data Module.
        """
        self.logger = logger
        self.sequence_length = sequence_length
        self.config = config
        self.test_size = test_size
        self.val_size = val_size
        self.preprocessor = DLDataPreprocessor(logger, sequence_length, config)
                
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Главный метод: выполняет Split, Normalization (fit/transform) и Sequencing.
        Гарантирует, что Test/Val данные не утекают в Normalization.
        """
        self.logger.info("DLDataModule: Запуск Train/Test Split.")
        
###         ### # ФИЛЬТРАЦИЯ: Оставляем только 'ch1_' фичи
###         ### ch1_cols = [col for col in df.columns if 'ch1_' in col and 'centered' in col]
        
        # 1. Train/Test Split (на уровне 2D DataFrame)
        # Мы используем X_tabular (фичи) и RUL (target)
        X_tabular = df.drop(columns=['RUL_hours'])
        y_vector = df['RUL_hours']
        
        # Split на Train/Test
        X_train_val, X_test_meta, y_train_val, y_test = train_test_split(
            X_tabular, y_vector, test_size=self.test_size, random_state=42, shuffle=True
        )
        
        # Split на Train/Validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size, random_state=42, shuffle=True
        )
        
        # 2. Normalization (Fit/Transform) и Sequencing
        # Обучаем Scaler ТОЛЬКО на Train
        self.preprocessor.fit(X_train)
        
        # Применяем Scaler и Sequencing
        
        # Train Set
        X_train_combined = X_train.copy()
        X_train_combined['RUL_hours'] = y_train
        X_train_3D, y_train_seq, _ = self.preprocessor.transform_to_sequences(X_train_combined)
        
        # Validation Set
        X_val_combined = X_val.copy()
        X_val_combined['RUL_hours'] = y_val
        X_val_3D, y_val_seq, _ = self.preprocessor.transform_to_sequences(X_val_combined)
        
        # Test Set
        X_test_combined = X_test_meta.copy()
        X_test_combined['RUL_hours'] = y_test
        X_test_3D, y_test_seq, meta_test = self.preprocessor.transform_to_sequences(X_test_combined)
        
        self.logger.info(f"DL Data Ready: Train: {X_train_3D.shape}, Val: {X_val_3D.shape}, Test: {X_test_3D.shape}")

        # Проверка на пустые массивы после секвенсирования - это критично
        if X_train_3D.shape[0] == 0:
            self.logger.error("Ошибка: Обучающий набор пуст после секвенсирования. Невозможно обучить модель.")
            raise ValueError("Обучающий набор пуст.")
        if X_val_3D.shape[0] == 0:
            self.logger.warning("Предупреждение: Валидационный набор пуст после секвенсирования. Обучение будет без валидации.")
            # Если валидационный набор пуст, нужно будет передать это в trainer,
            # чтобы он не пытался использовать validation_data.
            # Пока оставляем X_val_3D и y_val_seq пустыми, но корректно размерными,
            # чтобы trainer мог их принять (может быть заменен на None или отфильтрован в trainer).
        if X_test_3D.shape[0] == 0:
            self.logger.warning("Предупреждение: Тестовый набор пуст после секвенсирования. Оценка будет невозможна.")
            # Аналогично, если тестовый набор пуст

        return X_train_3D, y_train_seq, X_val_3D, y_val_seq, X_test_3D, y_test_seq, meta_test