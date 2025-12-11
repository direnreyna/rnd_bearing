# main.py

import io
import pandas as pd

import config
from src.dataset_builder import DatasetBuilder
from src.app_logger import AppLogger
from src.data_analyzer import DataAnalyzer
from src.advanced_analyzer import AdvancedDataAnalyzer
from src.spectral_analyzer import SpectralAnalyzer
from src.umap_visualizer import UMAPVisualizer

# Настраиваем pandas для более удобного вывода в консоль
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def main():
    """
    Главная функция-оркестратор.
    Запускает процесс создания или загрузки датасета и выводит результат.
    """
    logger = AppLogger.get_logger(__name__, config.LOG_FILEPATH)
    logger.info("Запуск MVP по предиктивному обслуживанию")

    # Шаг 1: Подготовка базового датасета
    builder = DatasetBuilder(raw_data_path=config.RAW_EXPERIMENT_DIR, processed_filepath=config.PROCESSED_DATA_FILEPATH, logger=logger)
    feature_df = builder.build_dataset()

    # Шаг 2: Базовый анализ и визуализация
    analyzer = DataAnalyzer(feature_df=feature_df, plots_dir=config.EDA_PLOTS_DIR, logger=logger, experiment_name=config.EXPERIMENT_NAME)
    analyzer.run()

    # Шаг 3: Продвинутый анализ
    adv_analyzer = AdvancedDataAnalyzer(plots_dir=config.EDA_PLOTS_DIR, extended_filepath=config.EXTENDED_DATA_FILEPATH, logger=logger, config=config)
    extended_df = adv_analyzer.run(feature_df)

    # Шаг 4: Извлечение спектральных признаков
    spectral_analyzer = SpectralAnalyzer(raw_data_path=config.RAW_EXPERIMENT_DIR, spectral_filepath=config.SPECTRAL_FEATURES_FILEPATH, logger=logger, config=config, window_size=config.WINDOW_SIZE, step=config.STEP, n_peaks=config.N_PEAKS, sampling_rate=config.SAMPLING_RATE)
    spectral_df = spectral_analyzer.run()

    # Шаг 5: Предобработка спектральных данных и UMAP визуализация
    umap_visualizer = UMAPVisualizer(output_path=config.UMAP_ANIMATION_FILEPATH, logger=logger, sample_fraction=config.UMAP_SAMPLE_FRACTION, animation_frequency=config.ANIMATION_FREQUENCY, animation_interval=config.ANIMATION_INTERVAL, experiment_name=config.EXPERIMENT_NAME, migration_window_days=config.MIGRATION_WINDOW_DAYS)
    # Запускаем визуализацию для каждого уровня производной
    umap_visualizer.run(spectral_df, derivative_level='d0')
    umap_visualizer.run(spectral_df, derivative_level='d1')
    umap_visualizer.run(spectral_df, derivative_level='d2')

    
    # Шаг 6: Вывод отладочной информации (если включено в конфиге)
    if config.DEBUG:
        logger.info("--- РЕЖИМ ОТЛАДКИ АКТИВИРОВАН ---")
        logger.info(f"Форма спектрального датасета: {spectral_df.shape}")
        logger.info(f"Первые 5 строк спектрального датасета:\n{spectral_df.head().to_string()}")
        
        with io.StringIO() as buffer:
            spectral_df.info(buf=buffer)
            info_str = buffer.getvalue()
            logger.info(f"\nИнформация о спектральном датасете:\n{info_str}")
        
        logger.info(f"Статистическая сводка по спектральному датасету:\n{spectral_df.describe().to_string()}")
        logger.info("--- КОНЕЦ СЕКЦИИ ОТЛАДКИ ---")
    
    logger.info("Пайплайн успешно завершен.")
    logger.info(f"Финальный датасет (расширенный) имеет форму: {processed_spectral_df.shape}")

if __name__ == "__main__":
    main()