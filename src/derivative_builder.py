# src/derivative_builder.py

import logging
import pandas as pd

class DerivativeBuilder:
    """
    Рассчитывает производные (скорость и ускорение) для временных рядов
    спектральных признаков.
    """

    def __init__(self, logger: logging.Logger):
        """
        Инициализирует строитель производных.

        Args:
            logger (logging.Logger): Экземпляр логгера.
        """
        self.logger = logger

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет в DataFrame колонки со скоростями и ускорениями.

        Args:
            df (pd.DataFrame): Входной DataFrame со спектральными признаками.

        Returns:
            pd.DataFrame: Обогащенный DataFrame с производными.
        """
        self.logger.info("Расчет производных (скоростей и ускорений)...")
        ### enriched_df = df.copy()

        # Находим все колонки с амплитудами, для которых будем считать производные
        amp_cols = [col for col in df.columns if '_amp' in col]
        if not amp_cols:
            self.logger.warning("Не найдены колонки с амплитудами ('_amp'). Пропускаем расчет производных.")
            return df

        # Группируем по подшипникам, чтобы производные не "перескакивали" между ними
        grouped = df.groupby('bearing') ## Работаем с исходным DF (df)

        derived_features = {}

        for col in amp_cols:
            # Рассчитываем скорость и заполняем NaN нулем
            derived_features[f'{col}_velo'] = grouped[col].transform(lambda x: x.diff()).fillna(0)
            
            # Рассчитываем ускорение и заполняем NaN нулем
            derived_features[f'{col}_accel'] = grouped[col].transform(lambda x: x.diff().diff()).fillna(0)

        # Однократное объединение всех производных признаков (быстрый способ)
        derived_df = pd.DataFrame(derived_features, index=df.index) 
        enriched_df = pd.concat([df, derived_df], axis=1) 

        self.logger.info(f"Успешно добавлено {len(amp_cols) * 2} колонок с производными.")
        return enriched_df