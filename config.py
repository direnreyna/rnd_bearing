# config.py

import pathlib

# ПУТИ
# Корень проекта, чтобы строить все остальные пути от него
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
# Путь к папке для обработанных данных (где будут лежать .parquet файлы)
PROCESSED_DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Создаем папку, если ее нет

# ПАРАМЕТРЫ ДАТАСЕТА
# Путь к каталогу с сырыми данными
RAW_DATA_SOURCE_DIR = pathlib.Path('/media/Cruiser/rnd_data/data')
# Имя эксперимента (папки), который мы обрабатываем
EXPERIMENT_NAME = '3rd_test'
# Полный путь к данным для текущего эксперимента
RAW_EXPERIMENT_DIR = RAW_DATA_SOURCE_DIR / EXPERIMENT_NAME

# ИМЕНА ФАЙЛОВ
# Имя файла для сохранения обработанного датафрейма
PROCESSED_DATA_FILENAME = f'{EXPERIMENT_NAME}_features.parquet'
# Полный путь к файлу с обработанными данными
PROCESSED_DATA_FILEPATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
    
# НАСТРОЙКИ ЛОГГЕРА
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True) # Создаем папку для логов
LOG_FILENAME = 'app.log'
LOG_FILEPATH = LOG_DIR / LOG_FILENAME 