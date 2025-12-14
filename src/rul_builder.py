# src/rul_builder.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

class RULBuilder:
    """
    Рассчитывает целевую переменную RUL (Remaining Useful Life) в часах.
    """

    def __init__(self, experiment_name: str, failure_map: Dict[str, Any], logger: logging.Logger, rul_min_threshold_hours: float):
        """
        Инициализирует RUL Builder.

        Args:
            experiment_name (str): Имя текущего эксперимента.
            failure_map (Dict[str, Any]): Карта подшипников, дошедших до отказа.
            logger (logging.Logger): Экземпляр логгера.
            rul_min_threshold_hours (float): Порог RUL для фильтрации "слишком здоровых" данных.
        """
        self.experiment_name = experiment_name
        self.failure_map = failure_map
        self.logger = logger
        self.failure_bearings = self.failure_map.get(self.experiment_name, [])
        self.rul_min_threshold_hours = rul_min_threshold_hours

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет колонку RUL_hours в DataFrame.

        Args:
            df (pd.DataFrame): Входной DataFrame со спектральными признаками.
                               Обязательно должен содержать колонки 'timestamp' (datetime) и 'bearing'.

        Returns:
            pd.DataFrame: Обогащенный DataFrame с RUL.
        """
        self.logger.info("Расчет целевой переменной RUL (Remaining Useful Life)...")
        enriched_df = df.copy()

        # 1. Определяем Time of Failure (TOA)
        # Так как RUL рассчитывается только для подшипников, дошедших до отказа, 
        # и все они ломаются в конце эксперимента, TOA - это последний таймстемп в данных.
        time_of_failure = enriched_df['timestamp'].max()
        self.logger.info(f"Общее время отказа (TOA) для всех отказавших подшипников: {time_of_failure}")
        self.logger.info(f"Подшипники, дошедшие до отказа: {self.failure_bearings}")

        # 2. Рассчитываем RUL для каждого подшипника
        def calculate_rul(row):
            bearing = row['bearing']
            current_time = row['timestamp']

            # Если подшипник дошел до отказа - рассчитываем RUL
            if bearing in self.failure_bearings:
                # RUL = (TOA - Текущее время) в часах
                rul_timedelta = time_of_failure - current_time
                return rul_timedelta.total_seconds() / 3600
            else:
                # Если подшипник "здоровый" (не сломался к концу эксперимента) - RUL = NaN
                return np.nan # Используем NA для лучшей совместимости с pandas

        enriched_df['RUL_hours'] = enriched_df.apply(calculate_rul, axis=1)

        #  ошибки LightGBM: Явное преобразование в float64
        enriched_df['RUL_hours'] = enriched_df['RUL_hours'].astype('float64')

        # 3. Добавляем фильтрацию (хотя это лучше делать на этапе ML, но для понимания)
        original_shape_before_all_filters = enriched_df.shape[0]
        # Удаляем здоровые подшипники, т.к. для них нет "жизни до отказа"
        enriched_df = enriched_df.dropna(subset=['RUL_hours'])
        nan_dropped_rows = original_shape_before_all_filters - enriched_df.shape[0]

        # 4. Применение RUL-фильтрации (T-min Cutoff)
        if self.rul_min_threshold_hours > 0:
            original_bearing_rows = enriched_df.shape[0]
            # Удаляем строки, где RUL слишком велик (слишком здоровое состояние)
            enriched_df = enriched_df[enriched_df['RUL_hours'] <= self.rul_min_threshold_hours]
            self.logger.info(f"RUL Cutoff (T-min): Удалено {original_bearing_rows - enriched_df.shape[0]} строк с RUL > {self.rul_min_threshold_hours} часов.")
        
        self.logger.info(f"RUL рассчитан. Удалено {nan_dropped_rows} строк (здоровые подшипники).")
        self.logger.info(f"Финальный датасет для обучения имеет форму: {enriched_df.shape}")
        
        return enriched_df