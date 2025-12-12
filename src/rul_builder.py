# src/rul_builder.py

import logging
import pandas as pd
from typing import Dict, Any

class RULBuilder:
    """
    Рассчитывает целевую переменную RUL (Remaining Useful Life) в часах.
    """

    def __init__(self, experiment_name: str, failure_map: Dict[str, Any], logger: logging.Logger):
        """
        Инициализирует RUL Builder.

        Args:
            experiment_name (str): Имя текущего эксперимента.
            failure_map (Dict[str, Any]): Карта подшипников, дошедших до отказа.
            logger (logging.Logger): Экземпляр логгера.
        """
        self.experiment_name = experiment_name
        self.failure_map = failure_map
        self.logger = logger
        self.failure_bearings = self.failure_map.get(self.experiment_name, [])

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
                return pd.NA # Используем NA для лучшей совместимости с pandas

        enriched_df['RUL_hours'] = enriched_df.apply(calculate_rul, axis=1)

        # 3. Добавляем фильтрацию (хотя это лучше делать на этапе ML, но для понимания)
        initial_shape = enriched_df.shape[0]
        # Удаляем здоровые подшипники, т.к. для них нет "жизни до отказа"
        enriched_df = enriched_df.dropna(subset=['RUL_hours'])
        final_shape = enriched_df.shape[0]

        self.logger.info(f"RUL рассчитан. Удалено {initial_shape - final_shape} строк (здоровые подшипники).")
        self.logger.info(f"Финальный датасет для обучения имеет форму: {enriched_df.shape}")
        
        return enriched_df