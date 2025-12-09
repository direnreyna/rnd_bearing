# main.py

import io
import pandas as pd

import config
from src.dataset_builder import DatasetBuilder
from src.app_logger import AppLogger

# Настраиваем pandas для более удобного вывода в консоль
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def main():
    """
    Главная функция-оркестратор.
    Запускает процесс создания или загрузки датасета и выводит результат.
    """
    # 1. Инициализируем логгер в самом начале
    logger = AppLogger.get_logger(__name__, config.LOG_FILEPATH)

    logger.info("Запуск MVP по предиктивному обслуживанию")

    # 2. Инициализируем строитель датасета с путями из конфига
    builder = DatasetBuilder(
        raw_data_path=config.RAW_EXPERIMENT_DIR,
        processed_filepath=config.PROCESSED_DATA_FILEPATH,
        logger=logger
    )

    # 3. Запускаем сборку (или загрузку из кэша)
    feature_df = builder.build_dataset()
    
    # 4. Выводим основную информацию о полученном датасете через логгер
    logger.info("--- Информация о датасете ---")
    logger.info(f"Форма датасета: {feature_df.shape}")
    
    # Для многострочного вывода используем to_string() или буфер
    logger.info(f"\nПервые 5 строк:\n{feature_df.head().to_string()}")

    with io.StringIO() as buffer:
        feature_df.info(buf=buffer)
        info_str = buffer.getvalue()
        logger.info(f"\nТипы данных и информация:\n{info_str}")

if __name__ == "__main__":
    main()