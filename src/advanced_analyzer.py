# src/advanced_analyzer.py

import logging
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AdvancedDataAnalyzer:
    """
    Выполняет продвинутый анализ данных, включая расчет
    синтетических признаков, таких как "Индекс Здоровья".
    """

    def __init__(self, plots_dir: pathlib.Path, extended_filepath: pathlib.Path, logger: logging.Logger, config):
        """
        Инициализирует анализатор.

        Args:
            plots_dir (pathlib.Path): Папка для сохранения графиков.
            extended_filepath (pathlib.Path): Путь к кэш-файлу с расширенными признаками.
            logger (logging.Logger): Экземпляр логгера.
            config: Модуль конфигурации для доступа к картам каналов.
        """
        self.plots_dir = plots_dir
        self.extended_filepath = extended_filepath
        self.logger = logger
        self.config = config
        sns.set(style="whitegrid")

    def run(self, base_feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Главный метод-оркестратор для этого класса.

        Выполняет всю цепочку: загрузка/расчет расширенных признаков,
        затем их визуализация.

        Args:
            base_feature_df (pd.DataFrame): Базовый датафрейм с признаками.

        Returns:
            pd.DataFrame: Расширенный датафрейм (загруженный из кэша или свежесозданный).
        """
        extended_df = self._get_or_create_extended_df(base_feature_df)
        self._plot_health_index(extended_df)
        return extended_df

    def _get_or_create_extended_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Внутренний метод, инкапсулирующий логику кэширования.
        """
        if self.extended_filepath.exists():
            self.logger.info(f"Загружаем расширенный датасет из кэша: {self.extended_filepath}")
            return pd.read_parquet(self.extended_filepath)
        
        self.logger.info("Расширенный датасет не найден. Создаем новый...")
        extended_df = self._calculate_health_index(df)
        self.logger.info(f"Сохраняем расширенный датасет в файл: {self.extended_filepath}")
        extended_df.to_parquet(self.extended_filepath)
        return extended_df

    def _calculate_health_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает "Индекс Здоровья" (Health Index) для каждого подшипника
        с использованием метода главных компонент (PCA).

        Args:
            df (pd.DataFrame): Входной датафрейм с базовыми признаками.

        Returns:
            pd.DataFrame: Расширенный датафрейм с добавленными колонками Health Index.
        """
        self.logger.info("Расчет 'Индекса Здоровья' с помощью PCA...")
        extended_df = df.copy()
        
        # Получаем карту каналов для текущего эксперимента
        experiment_name = self.extended_filepath.stem.split('_')[0] + '_test'
        channel_map = self.config.EXPERIMENT_CHANNELS[experiment_name]

        for bearing_name in channel_map.keys():        
            # 1. Выбираем все признаки для текущего подшипника
            feature_cols = [col for col in df.columns if bearing_name in col]
            bearing_features = df[feature_cols]

            # 2. Стандартизируем признаки (важно для PCA)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(bearing_features)

            # 3. Применяем PCA для снижения размерности до 1 компоненты
            pca = PCA(n_components=1)
            health_index = pca.fit_transform(scaled_features)
            
            # 4. Корректируем направление индекса.
            # Знак главной компоненты произволен. Мы хотим, чтобы рост
            # индекса всегда означал деградацию. Проверяем корреляцию
            # с временным индексом (предполагаем, что деградация со временем растет).
            # Если корреляция отрицательная - инвертируем знак.
            time_numeric = (df.index - df.index.min()).total_seconds()
            correlation = pd.Series(health_index.flatten()).corr(pd.Series(time_numeric))
            
            if correlation < 0:
                health_index = -health_index # Инвертируем

            # 5. Добавляем новую колонку в датафрейм
            extended_df[f'{bearing_name}_health_index'] = health_index
            self.logger.info(f"Индекс здоровья для {bearing_name} успешно рассчитан.")
            
        return extended_df

    def _plot_health_index(self, df: pd.DataFrame):
        """
        Строит и сохраняет график "Индекса Здоровья" для всех подшипников.

        Args:
            df (pd.DataFrame): Расширенный датафрейм с колонками Health Index.
        """
        self.logger.info("Создание графика 'Индекса Здоровья'...")
        
        health_index_cols = [col for col in df.columns if 'health_index' in col]
        
        plt.figure(figsize=(15, 8))
        
        for col in health_index_cols:
            # Сглаживаем для лучшей визуализации тренда
            plt.plot(df.index, df[col].rolling(window=200).mean(), label=col)
        
        plt.title('Синтетический "Индекс Здоровья" (Health Index) во времени (Окно=200)', fontsize=16)
        plt.xlabel('Дата и время', fontsize=12)
        plt.ylabel('Значение Health Index', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        save_path = self.plots_dir / 'health_index_trend.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График 'Индекса Здоровья' сохранен в: {save_path}")
