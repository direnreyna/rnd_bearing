# src/dataset_builder.py

import os
import sys
import pathlib
from typing import Dict, List
import logging

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tqdm import tqdm


class DatasetBuilder:
    """Собирает датасет с признаками из сырых файлов эксперимента."""

    def __init__(self, raw_data_path: pathlib.Path, processed_filepath: pathlib.Path, logger: logging.Logger):
        """Инициализирует сборщик датасета.

        Args:
            raw_data_path (pathlib.Path): Путь к папке с сырыми файлами эксперимента.
            processed_filepath (pathlib.Path): Путь для сохранения обработанного файла.
            logger (logging.Logger): Экземпляр логгера для записи событий.
        """
        self.raw_data_path = raw_data_path
        self.processed_filepath = processed_filepath
        self.logger = logger

    def build_dataset(self) -> pd.DataFrame:
        """
        Главный метод. Проверяет наличие кэша, если его нет - создает датасет.

        Returns:
            pd.DataFrame: Готовый датафрейм с рассчитанными признаками.
        """
        if self.processed_filepath.exists():
            self.logger.info(f"Загружаем обработанный датасет из кэша: {self.processed_filepath}")
            return pd.read_parquet(self.processed_filepath)
        
        self.logger.info("Обработанный датасет не найден. Создаем новый...")
        return self._create_feature_dataset()

    def _create_feature_dataset(self) -> pd.DataFrame:
        """
        Создает датасет путем итерации по всем файлам и извлечения признаков.
        """
        # Получаем отсортированный список файлов, чтобы сохранить временную последовательность
        file_list = sorted([f for f in self.raw_data_path.iterdir() if f.is_file()])
        
        # Используем tqdm для наглядного прогресс-бара
        features_list = [
            self._extract_features_from_file(file_path) 
            for file_path in tqdm(file_list, desc="Обработка файлов", file=sys.stdout)
        ]
        
        # Собираем все признаки в один датафрейм
        feature_df = pd.DataFrame(features_list)
        # Устанавливаем таймстемп в качестве индекса
        feature_df = feature_df.set_index('timestamp')
        
        self.logger.info(f"Сохраняем датасет в файл: {self.processed_filepath}")
        feature_df.to_parquet(self.processed_filepath)
        
        return feature_df

    def _extract_features_from_file(self, file_path: pathlib.Path) -> Dict:
        """
        Извлекает признаки из одного файла с данными вибрации.

        Args:
            file_path (pathlib.Path): Путь к файлу.

        Returns:
            Dict: Словарь с извлеченными признаками.
        """
        # Загружаем данные. У нас нет заголовков, разделитель - табуляция.
        raw_df = pd.read_csv(file_path, sep='\t', header=None)
        
        # Создаем имена колонок в зависимости от их количества
        num_columns = raw_df.shape[1]
        raw_df.columns = [f'bearing_{i+1}' for i in range(num_columns)]
        
        features = {}
        # Извлекаем таймстемп из имени файла
        timestamp_str = file_path.stem.replace('.', '-', 3).replace('.', ':')
        features['timestamp'] = pd.to_datetime(timestamp_str)

        # Рассчитываем признаки для каждого подшипника (каждой колонки)
        for col in raw_df.columns:
            signal = raw_df[col]
            
            # Статистические моменты
            features[f'{col}_mean'] = signal.mean()
            features[f'{col}_std'] = signal.std()
            features[f'{col}_skew'] = skew(signal)
            features[f'{col}_kurtosis'] = kurtosis(signal) # Fisher's kurtosis
            
            # Амплитудные характеристики
            features[f'{col}_rms'] = np.sqrt(np.mean(signal**2))
            peak = signal.abs().max()
            features[f'{col}_peak'] = peak
            # Крест-фактор. Добавляем малое число, чтобы избежать деления на ноль
            features[f'{col}_crest_factor'] = peak / (features[f'{col}_rms'] + 1e-6)

        return features