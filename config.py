# config.py

import pathlib

# ОСНОВНЫЕ ПУТИ
# Корень проекта, чтобы строить все остальные пути от него
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
# Путь к папке для обработанных данных (где будут лежать .parquet файлы)
PROCESSED_DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Создаем папку, если ее нет

# ПАРАМЕТРЫ ДАТАСЕТА
# Путь к каталогу с сырыми данными
RAW_DATA_SOURCE_DIR = pathlib.Path('/media/Cruiser/rnd_data/data')
# Имя эксперимента (папки), который мы обрабатываем
EXPERIMENT_NAME = '1st_test' ## '1st_test' / '2nd_test' / '3rd_test'
# Полный путь к данным для текущего эксперимента
RAW_EXPERIMENT_DIR = RAW_DATA_SOURCE_DIR / EXPERIMENT_NAME

# ИМЕНА ФАЙЛОВ
# Имя файла для сохранения обработанного датафрейма
PROCESSED_DATA_FILENAME = f'{EXPERIMENT_NAME}_features.parquet'
# Полный путь к файлу с обработанными данными
PROCESSED_DATA_FILEPATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
# Имя файла для сохранения расширенного датафрейма с Health Index
EXTENDED_DATA_FILENAME = f'{EXPERIMENT_NAME}_extended_features.parquet'
# Полный путь к файлу с расширенными данными
EXTENDED_DATA_FILEPATH = PROCESSED_DATA_DIR / EXTENDED_DATA_FILENAME
# Имена файлов для спектрального анализа
SPECTRAL_FEATURES_FILENAME = f'{EXPERIMENT_NAME}_spectral_features.parquet'
SPECTRAL_FEATURES_FILEPATH = PROCESSED_DATA_DIR / SPECTRAL_FEATURES_FILENAME

# ФАЙЛ ДЛЯ СОХРАНЕНИЯ ОБУЧЕННОЙ МОДЕЛИ
MODEL_FILENAME = f'{EXPERIMENT_NAME}_lgbm_rul_model.joblib'
MODEL_FILEPATH = PROCESSED_DATA_DIR / MODEL_FILENAME

# ФОРМАТ ВРЕМЕННОЙ МЕТКИ (для парсинга из имени файла)
# Пример: 2003.10.22.12.06.24 -> %Y.%m.%d.%H.%M.%S
TIMESTAMP_FORMAT = '%Y.%m.%d.%H.%M.%S'

# Описание структуры экспериментов
EXPERIMENT_CHANNELS = {
    '1st_test': {
        'bearing_1': [0, 1],  # Колонки 0 (X) и 1 (Y)
        'bearing_2': [2, 3],
        'bearing_3': [4, 5],
        'bearing_4': [6, 7]
    },
    '2nd_test': {
        'bearing_1': [0], # Только колонка 0
        'bearing_2': [1],
        'bearing_3': [2],
        'bearing_4': [3]
    },
    '3rd_test': {
        'bearing_1': [0],
        'bearing_2': [1],
        'bearing_3': [2],
        'bearing_4': [3]
    }
}

# СПЕКТРАЛЬНЫЙ АНАЛИЗ
WINDOW_SIZE = 4096      # Размер окна для FFT
STEP = 512              # Шаг, с которым двигается окно (создает перекрытие)
N_PEAKS = 10            # Количество самых сильных частотных пиков для извлечения
SAMPLING_RATE = 20000   # Частота дискретизации в Гц (из документации)

# ЛОГГЕР
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True) # Создаем папку для логов
LOG_FILENAME = 'app.log'
LOG_FILEPATH = LOG_DIR / LOG_FILENAME 

# EDA
# Пути для результатов анализа
EDA_PLOTS_DIR = ROOT_DIR / 'plots'
EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True) # Создаем папку для графиков

# DEBUG-отладка
DEBUG = True # Включить/выключить вывод детальной информации о датафреймах

# UMAP визуализация
UMAP_ANIMATION_FILENAME = f'{EXPERIMENT_NAME}_umap_evolution.gif'
UMAP_ANIMATION_FILEPATH = EDA_PLOTS_DIR / UMAP_ANIMATION_FILENAME
UMAP_SAMPLE_FRACTION = 1.0 # Какую долю последних данных использовать для UMAP (1.0 = все данные)
# УПРАВЛЕНИЕ АНИМАЦИЕЙ (отключение создания)
ENABLE_UMAP_GIFS = False # Если False, то Шаг 8 будет пропущен (экономия времени).

# ТЮНИНГ МОДЕЛИ
ENABLE_MODEL_TUNING = False # Включить/выключить тюнинг (RandomizedSearchCV)
# Параметры для RandomizedSearchCV
LGBM_TUNING_PARAMS = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 50],
    'max_depth': [-1, 5, 10],
    'min_child_samples': [10, 20, 30]
}
N_ITER_SEARCH = 10 # Количество итераций для RandomizedSearchCV

# Единица измерения частоты кадров: 'D' - день, 'H' - час, 'T' или 'min' - минута
ANIMATION_FREQUENCY = 'D'
# КАРТА ПОДШИПНИКОВ, ДОШЕДШИХ ДО ОТКАЗА (TOA)
# Ключ: Имя эксперимента. Значение: Список подшипников, для которых RUL рассчитывается до последнего таймстемпа.
FAILURE_BEARINGS_MAP = {
    '1st_test': ['bearing_3', 'bearing_4'],
    '2nd_test': ['bearing_1'],
    '3rd_test': ['bearing_3']
}
# Интервал: сколько единиц частоты в одном кадре. 
# 'H' и 6 = 1 кадр каждые 6 часов. 'T' и 10 = 1 кадр каждые 10 минут.
ANIMATION_INTERVAL = 1
MIGRATION_WINDOW_DAYS = 3 # Длина "шлейфа" в днях для миграционной гифки