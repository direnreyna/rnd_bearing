# src/data_analyzer.py

import logging
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    """Выполняет исследовательский анализ данных (EDA) и визуализацию."""

    def __init__(self, feature_df: pd.DataFrame, plots_dir: pathlib.Path, logger: logging.Logger):
        """
        Инициализирует анализатор.

        Args:
            feature_df (pd.DataFrame): Датафрейм с признаками.
            plots_dir (pathlib.Path): Папка для сохранения графиков.
            logger (logging.Logger): Экземпляр логгера.
        """
        self.df = feature_df
        self.plots_dir = plots_dir
        self.logger = logger
        # Устанавливаем стиль для графиков
        sns.set(style="whitegrid")

    def run(self):
        """
        Строит и сохраняет графики ключевых признаков для всех подшипников.
        """
        self.logger.info("Создание графиков ключевых признаков...")
        
        # Признаки, которые мы хотим визуализировать
        features_to_plot = ['rms', 'kurtosis', 'crest_factor']
        
        for feature in features_to_plot:
            self._plot_feature_for_all_bearings(feature)
            
        self.logger.info(f"Графики базового EDA сохранены в папку: {self.plots_dir}")

    def _plot_feature_for_all_bearings(self, feature_name: str):
        """
        Вспомогательный метод для построения графика одного признака.

        Args:
            feature_name (str): Базовое имя признака (например, 'rms').
        """
        plt.figure(figsize=(15, 8))
        
        # Собираем колонки для текущего признака (bearing_1_rms, bearing_2_rms, ...)
        feature_columns = [col for col in self.df.columns if feature_name in col]
        
        for col in feature_columns:
            # Используем rolling window для сглаживания графика, чтобы лучше видеть тренд
            # Размер окна (200) можно подобрать, он означает "среднее по 200 точкам"
            plt.plot(self.df.index, self.df[col].rolling(window=200).mean(), label=col)
            
        plt.title(f'Сглаженный показатель "{feature_name.upper()}" во времени (Окно=200)', fontsize=16)
        plt.xlabel('Дата и время', fontsize=12)
        plt.ylabel(f'Значение {feature_name}', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # Сохраняем график в файл
        save_path = self.plots_dir / f'{feature_name}_trend.png'
        plt.savefig(save_path)
        plt.close() # Закрываем фигуру, чтобы не отображать ее в блокноте и освободить память