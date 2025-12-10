# main.py

import io
import pandas as pd

import config
from src.dataset_builder import DatasetBuilder
from src.app_logger import AppLogger
from src.data_analyzer import DataAnalyzer
from src.advanced_analyzer import AdvancedDataAnalyzer

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
    analyzer = DataAnalyzer(feature_df=feature_df, plots_dir=config.EDA_PLOTS_DIR, logger=logger)
    analyzer.run()

    # Шаг 3: Продвинутый анализ
    adv_analyzer = AdvancedDataAnalyzer(plots_dir=config.EDA_PLOTS_DIR, extended_filepath=config.EXTENDED_DATA_FILEPATH, logger=logger)
    extended_df = adv_analyzer.run(feature_df)

    logger.info("Пайплайн успешно завершен.")
    logger.info(f"Финальный датасет (расширенный) имеет форму: {extended_df.shape}")
    
    ### builder = DatasetBuilder(raw_data_path=config.RAW_EXPERIMENT_DIR, processed_filepath=config.PROCESSED_DATA_FILEPATH, logger=logger)
    ### 
    ### # Загружаем датасет
    ### feature_df = builder.build_dataset()
    ### logger.info(f"Форма датасета: {feature_df.shape}")
    ### 
    ### # Для многострочного вывода используем to_string() или буфер
    ### # logger.info(f"\nПервые 5 строк:\n{feature_df.head().to_string()}")
    ### 
    ### with io.StringIO() as buffer:
    ###     feature_df.info(buf=buffer)
    ###     info_str = buffer.getvalue()
    ###     logger.info(f"\nТипы данных и информация:\n{info_str}")
    ### 
    ### # Расчет или загрузка "Индекса Здоровья"
    ### logger.info("Продвинутый анализ данных (Health Index)")
    ### adv_analyzer = AdvancedDataAnalyzer(plots_dir=config.EDA_PLOTS_DIR, logger=logger)
    ### 
    ### if config.EXTENDED_DATA_FILEPATH.exists():
    ###     logger.info(f"Загружаем расширенный датасет из кэша: {config.EXTENDED_DATA_FILEPATH}")
    ###     extended_df = pd.read_parquet(config.EXTENDED_DATA_FILEPATH)
    ### else:
    ###     logger.info("Расширенный датасет не найден. Создаем новый...")
    ###     extended_df = adv_analyzer.calculate_health_index(feature_df)
    ###     logger.info(f"Сохраняем расширенный датасет в файл: {config.EXTENDED_DATA_FILEPATH}")
    ###     extended_df.to_parquet(config.EXTENDED_DATA_FILEPATH)
    ### 
    ### # 5. Запускаем исследовательский анализ данных (EDA)
    ### logger.info("Исследовательский анализ данных (EDA)")
    ### analyzer = DataAnalyzer(
    ###     feature_df=feature_df,
    ###     plots_dir=config.EDA_PLOTS_DIR,
    ###     logger=logger
    ### )
    ### analyzer.plot_key_features()
    ### logger.info(f"Графики EDA сохранены в папку: {config.EDA_PLOTS_DIR}")
    ### 
    ### # Визуализация "Индекса Здоровья"
    ### logger.info("Визуализация продвинутого анализа (Health Index)")
    ### adv_analyzer.plot_health_index(extended_df)
    ### logger.info(f"График Health Index сохранен в папку: {config.EDA_PLOTS_DIR}")

if __name__ == "__main__":
    main()