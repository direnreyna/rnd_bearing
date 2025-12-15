# main.py

import io
import pandas as pd

import joblib
import mlflow
import config
from src.dataset_builder import DatasetBuilder
from src.app_logger import AppLogger
from src.data_analyzer import DataAnalyzer
from src.advanced_analyzer import AdvancedDataAnalyzer
from src.spectral_analyzer import SpectralAnalyzer
from src.baseline_transformer import BaselineTransformer
from src.rul_builder import RULBuilder
from src.tabular_model_trainer import TabularModelTrainer
from src.model_evaluator import ModelEvaluator
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

    ### # Подготовка базового датасета
    ### builder = DatasetBuilder(raw_data_path=config.RAW_EXPERIMENT_DIR, processed_filepath=config.PROCESSED_DATA_FILEPATH, logger=logger)
    ### feature_df = builder.build_dataset()
    ### 
    ### # Базовый анализ и визуализация (EDA)
    ### analyzer = DataAnalyzer(feature_df=feature_df, plots_dir=config.EDA_PLOTS_DIR, logger=logger, experiment_name=config.EXPERIMENT_NAME)
    ### analyzer.run()
    ### 
    ### # Продвинутый анализ (EDA-2)
    ### adv_analyzer = AdvancedDataAnalyzer(plots_dir=config.EDA_PLOTS_DIR, extended_filepath=config.EXTENDED_DATA_FILEPATH, logger=logger, config=config)
    ### extended_df = adv_analyzer.run(feature_df)

    # Настройка MLflow Tracking
    if config.ENABLE_MLFLOW_TRACKING:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow настроен. Tracking URI: {config.MLFLOW_TRACKING_URI}")
        logger.info(f"MLflow Experiment: {config.MLFLOW_EXPERIMENT_NAME}")
    
    # Извлечение спектральных признаков
    spectral_analyzer = SpectralAnalyzer(raw_data_path=config.RAW_EXPERIMENT_DIR, spectral_filepath=config.SPECTRAL_FEATURES_FILEPATH, logger=logger, config=config, window_size=config.WINDOW_SIZE, step=config.STEP, n_peaks=config.N_PEAKS, sampling_rate=config.SAMPLING_RATE)
    spectral_df = spectral_analyzer.run()

    # Трансформация признаков (Baseline Centering)
    baseline_transformer = BaselineTransformer(logger=logger, baseline_windows_count=config.BASELINE_WINDOWS_COUNT)
    transformed_df = baseline_transformer.run(spectral_df)

    # Создание целевой переменной (RUL)
    rul_builder = RULBuilder(experiment_name=config.EXPERIMENT_NAME, failure_map=config.FAILURE_BEARINGS_MAP, logger=logger, rul_min_threshold_hours=config.RUL_MIN_THRESHOLD_HOURS, target_failure_bearing_index=config.TARGET_FAILURE_BEARING_INDEX, nominal_window_interval_minutes=config.NOMINAL_WINDOW_INTERVAL_MINUTES)
    processed_spectral_df = rul_builder.run(transformed_df)

    # Создание и обучение базовой модели (Baseline)
    trainer = TabularModelTrainer(logger=logger, experiment_name=config.EXPERIMENT_NAME)
    # Возвращаем все результаты для визуализации оценки
    model, X_test_meta, y_test_train, y_pred_train, feature_importance = trainer.run(processed_spectral_df)

    # Оценка и визуализация результатов модели
    evaluator = ModelEvaluator(plots_dir=config.EDA_PLOTS_DIR, logger=logger)
    evaluator.run(X_test_meta, y_test_train, y_pred_train, feature_importance)

    # СЕКЦИЯ: Перекрестный эксперимент проверки модели на одних данных обученной на других данных
    if config.ENABLE_CROSS_EXPERIMENT_PREDICTION:
        logger.info(f"Запуск Cross-Experiment Prediction: Тестирование {config.EXPERIMENT_NAME} на {config.TARGET_TEST_EXPERIMENT_NAME}")
        
        # 1. Формируем пути для целевого эксперимента
        test_exp_name = config.TARGET_TEST_EXPERIMENT_NAME
        raw_test_dir = config.RAW_DATA_SOURCE_DIR / test_exp_name
        spectral_test_filepath = config.PROCESSED_DATA_DIR / f'{test_exp_name}_spectral_features.parquet'
        
        # 2. Повторяем Feature Engineering и RUL-расчет для тестового эксперимента
        spectral_analyzer_test = SpectralAnalyzer(raw_data_path=raw_test_dir, spectral_filepath=spectral_test_filepath, logger=logger, config=config, window_size=config.WINDOW_SIZE, step=config.STEP, n_peaks=config.N_PEAKS, sampling_rate=config.SAMPLING_RATE)
        spectral_df_test = spectral_analyzer_test.run()

        baseline_transformer_test = BaselineTransformer(logger=logger, baseline_windows_count=config.BASELINE_WINDOWS_COUNT)
        transformed_df_test = baseline_transformer_test.run(spectral_df_test)

        rul_builder_test = RULBuilder(experiment_name=test_exp_name, failure_map=config.FAILURE_BEARINGS_MAP, logger=logger, rul_min_threshold_hours=0, target_failure_bearing_index=config.TARGET_FAILURE_BEARING_INDEX, nominal_window_interval_minutes=config.NOMINAL_WINDOW_INTERVAL_MINUTES)
        processed_spectral_df_test = rul_builder_test.run(transformed_df_test)
        
        # 3. Подготовка тестовых данных (X, y)
        X_test_meta_pred, y_test_pred = trainer._prepare_predict_data(processed_spectral_df_test) 
        
        # 4. Предсказание и оценка
        X_test_model_pred = X_test_meta_pred.drop(columns=['timestamp', 'bearing'])
        
        if X_test_model_pred.shape[0] > 0:
            # Загружаем модель (хотя в памяти уже есть, но для Production-имитации)
            model_pred = joblib.load(config.MODEL_FILEPATH)
            
            y_pred_test = model_pred.predict(X_test_model_pred)
            y_pred_series_test = pd.Series(y_pred_test, index=y_test_pred.index) # Удален dtype='float64'
            
            # Логгирование метрик и визуализация
            evaluator.run(X_test_meta_pred, y_test_pred, y_pred_series_test, feature_importance)
            logger.info(f"Cross-Experiment Prediction завершен. Результаты сохранены.")
        else:
            logger.warning("Тестовый датасет для предсказания пуст после фильтрации. Шаг пропущен.")

    # Предобработка спектральных данных и UMAP визуализация
    if config.ENABLE_UMAP_GIFS: 
        umap_visualizer = UMAPVisualizer(output_path=config.UMAP_ANIMATION_FILEPATH, logger=logger, sample_fraction=config.UMAP_SAMPLE_FRACTION, animation_frequency=config.ANIMATION_FREQUENCY, animation_interval=config.ANIMATION_INTERVAL, experiment_name=config.EXPERIMENT_NAME, migration_window_days=config.MIGRATION_WINDOW_DAYS)
        logger.info("Запуск UMAP визуализации (включено в конфиге).")
        # Запускаем визуализацию для каждого уровня производной
        umap_visualizer.run(processed_spectral_df, derivative_level='d0')
        umap_visualizer.run(processed_spectral_df, derivative_level='d1')
        umap_visualizer.run(processed_spectral_df, derivative_level='d2')
    else:
        logger.warning("UMAP визуализация отключена в конфиге (ENABLE_UMAP_GIFS=False). Шаг пропущен.")

    # Вывод отладочной информации (если включено в конфиге)
    if config.DEBUG:
        with io.StringIO() as buffer:
            processed_spectral_df.info(buf=buffer)
            info_str = buffer.getvalue()
            logger.info(f"\nИнформация о спектральном датасете:\n{info_str}")

        # Дополнительная отладочная информация по модели
        logger.info("--- ОТЛАДКА МОДЕЛИ BASELINE ---")
        logger.info(f"ТОП-10 самых важных признаков:\n{feature_importance.head(10).to_string()}")
        logger.info("--- КОНЕЦ СЕКЦИИ ОТЛАДКИ МОДЕЛИ ---")
    
    logger.info("Пайплайн успешно завершен.")
    logger.info(f"Финальный датасет (расширенный) имеет форму: {processed_spectral_df.shape}")

if __name__ == "__main__":
    main()