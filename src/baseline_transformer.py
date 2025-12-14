# src/baseline_transformer.py

import logging
import pandas as pd
from typing import Dict, Any

class BaselineTransformer:
    """
    Создает новые признаки путем центрирования (вычитания) начального
    "здорового" среднего значения для каждого подшипника.
    """

    def __init__(self, logger: logging.Logger, baseline_windows_count: int):
        """
        Инициализирует трансформер.

        Args:
            logger (logging.Logger): Экземпляр логгера.
            baseline_windows_count (int): Количество начальных окон для расчета Baseline.
        """
        self.logger = logger
        self.baseline_windows_count = baseline_windows_count

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает центрированные признаки.

        Args:
            df (pd.DataFrame): Входной DataFrame со спектральными признаками.

        Returns:
            pd.DataFrame: DataFrame с центрированными признаками.
        """
        self.logger.info(f"Запуск трансформации Baseline Centering (N={self.baseline_windows_count} окон)...")


        # 0. Сбрасываем индекс, чтобы 'timestamp', 'bearing', 'window_id' стали обычными колонками
        # Это предотвращает ошибки с MultiIndex при группировке.
        transformed_df = df.reset_index(drop=True).copy()
        
        # 1. Определяем, какие колонки центрировать
        # Центрируем только амплитуды (amp) и их производные (velo, accel)
        feature_cols = [col for col in transformed_df.columns if '_amp' in col]

        if not feature_cols:
            self.logger.warning("Не найдены амплитудные признаки ('_amp'). Пропускаем центрирование.")
            return transformed_df

        # 2. Расчет Baseline (среднее значение первых N окон для каждого подшипника)
        self.logger.info("Расчет Baseline (среднее значение) по подшипникам...")

        # A. Вычисляем среднее только для первых N строк каждой группы
        df_baseline = transformed_df.groupby('bearing').head(self.baseline_windows_count)
        baseline_values = (
            df_baseline.groupby('bearing')[feature_cols]
            .mean()
        )

        self.logger.info(f"Рассчитан Baseline для {len(baseline_values)} подшипников.")

        # 3. Применение центрирования и замена исходных колонок
        centered_data = {}

        for col in feature_cols:
            new_col_name = f"{col}_centered"
            
            # Применяем вычитание Baseline для каждого подшипника
            def center_column(series):
                bearing_name = series.name # Имя подшипника
                if bearing_name in baseline_values.index:
                    baseline = baseline_values.loc[bearing_name, col]
                    return series - baseline
                return series # Если нет Baseline, оставляем как есть

            # Создаем новую колонку с центрированными значениями
            centered_series = transformed_df.groupby('bearing')[col].transform(center_column)
            
            # Добавляем серию в словарь вместо DataFrame
            centered_data[new_col_name] = centered_series

            ### # Добавляем новую колонку
            ### transformed_df[new_col_name] = centered_series
            ### 
            ### # Удаляем старую колонку сразу после создания новой
            ### transformed_df = transformed_df.drop(columns=[col], errors='ignore')

        ### # Важно: Слияние по индексу. centered_features уже имеет правильный индекс.
        ### transformed_df = transformed_df.merge(centered_features, left_index=True, right_index=True, how='left')

        # 4. Удаление исходных фич (amp, velo, accel) и добавление центрированных
        transformed_df = transformed_df.drop(columns=feature_cols, errors='ignore')

        # 5. Объединение за одну операцию (решение PerformanceWarning)
        transformed_df = pd.concat([transformed_df, pd.DataFrame(centered_data, index=transformed_df.index)], axis=1)

        self.logger.info(f"Трансформация Baseline Centering завершена. Добавлено {len(feature_cols)} новых признаков.")
        return transformed_df