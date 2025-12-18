# main.py

import io
import pandas as pd
import numpy as np
import argparse

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
from src.dl_data_preprocessor import DLDataPreprocessor
from src.dl_data_module import DLDataModule
from src.dl_model_trainer import DLModelTrainer

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

    parser = argparse.ArgumentParser(description="Пайплайн предиктивного обслуживания подшипников.")
    parser.add_argument("--mode", type=str, default=None, # По умолчанию None, будет train_and_tune
                        choices=["finetune", "inference"], # train_and_tune не указываем здесь
                        help="Режим работы: 'finetune' (дообучение с весами), 'inference' (только предсказание)."
                             " Если не указан, по умолчанию используется 'train_and_tune'.")
    args = parser.parse_args()

    # Определяем фактический режим работы
    current_mode = args.mode if args.mode else "train_and_tune"

    logger.info(f"Запуск в режиме: {current_mode.upper()}")

    processed_spectral_df = pd.DataFrame()
    feature_importance = pd.Series(dtype='float64')

    # Инициализация Evaluator до условных блоков для типов моделей
    evaluator = ModelEvaluator(plots_dir=config.EDA_PLOTS_DIR, logger=logger, model_type=config.MODEL_TYPE)
    
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

    #################################################################################
    # ОРКЕСТРАЦИЯ: DL / Tabular
    #################################################################################
    if config.MODEL_TYPE == 'LGBM' or config.MODEL_TYPE == 'CATB':
        logger.info(f"Запуск Tabular Пайплайна ({config.MODEL_TYPE}).")

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
    
    #################################################################################
    elif config.MODEL_TYPE == 'DL':
    #################################################################################
        logger.info("Запуск Deep Learning Пайплайна.")
        
        # 1. Feature Engineering (Используем ту же логику для извлечения табличных фичей)
        spectral_analyzer = SpectralAnalyzer(raw_data_path=config.RAW_EXPERIMENT_DIR, spectral_filepath=config.SPECTRAL_FEATURES_FILEPATH, logger=logger, config=config, window_size=config.WINDOW_SIZE, step=config.STEP, n_peaks=config.N_PEAKS, sampling_rate=config.SAMPLING_RATE)
        spectral_df = spectral_analyzer.run()

        baseline_transformer = BaselineTransformer(logger=logger, baseline_windows_count=config.BASELINE_WINDOWS_COUNT)
        transformed_df = baseline_transformer.run(spectral_df)

        rul_builder = RULBuilder(experiment_name=config.EXPERIMENT_NAME, failure_map=config.FAILURE_BEARINGS_MAP, logger=logger, rul_min_threshold_hours=config.RUL_MIN_THRESHOLD_HOURS, target_failure_bearing_index=config.TARGET_FAILURE_BEARING_INDEX, nominal_window_interval_minutes=config.NOMINAL_WINDOW_INTERVAL_MINUTES)
        processed_spectral_df = rul_builder.run(transformed_df)
        

        # Оркестрация DL-пайплайна в зависимости от режима работы
        #=========================================================
        if current_mode == "train_and_tune":
        #=========================================================
            logger.info("Режим: ОБУЧЕНИЕ И ТЮНИНГ DL-МОДЕЛИ.")

            # 2. Подготовка DL-данных (Sequencing/Split/Normalize)
            dl_data_module = DLDataModule(
                logger=logger,
                sequence_length=config.DL_SEQUENCE_LENGTH,
                config=config
            )
            X_train_3D, y_train_seq, X_val_3D, y_val_seq, X_test_3D, y_test_seq, X_test_meta_dl = dl_data_module.prepare_data(processed_spectral_df)

            # 3. Обучение DL-модели
            dl_trainer = DLModelTrainer(
                logger=logger, 
                experiment_name=config.EXPERIMENT_NAME, 
                sequence_length=config.DL_SEQUENCE_LENGTH, 
                lstm_units=config.DL_LSTM_UNITS, 
                epochs=config.DL_EPOCHS, 
                batch_size=config.DL_BATCH_SIZE
            )

            model = dl_trainer.run(X_train_3D, y_train_seq, X_val_3D, y_val_seq)

            # 4. Оценка и визуализация (Нужно адаптировать Evaluator для DL)
            # Выходы DL-моделей - это чистые массивы numpy
            y_pred = dl_trainer.predict(X_test_3D)

            # DL-модели не имеют feature_importance в Tabular-стиле. Присваиваем пустую Series.
            feature_importance = pd.Series(data=[1.0], index=['DL_MODEL'], dtype='float64')

            logger.warning("DL-ОЦЕНКА: Требуется дальнейшая адаптация Evaluator для 3D-данных.")
            evaluator.run(X_test_meta_dl, pd.Series(y_test_seq, index=X_test_meta_dl.index), pd.Series(y_pred, index=X_test_meta_dl.index), feature_importance)

        #=========================================================
        elif current_mode == "finetune":
        #=========================================================
            logger.info("Режим: ДООБУЧЕНИЕ (FINETUNE) DL-МОДЕЛИ.")

            # 2. Подготовка DL-данных (Sequencing/Split/Normalize) - для дообучения может потребоваться свой набор
            dl_data_module = DLDataModule(
                logger=logger,
                sequence_length=config.DL_SEQUENCE_LENGTH,
                config=config
            )
            # Для дообучения мы берем весь processed_spectral_df (или его часть)
            X_train_3D, y_train_seq, X_val_3D, y_val_seq, X_test_3D, y_test_seq, X_test_meta_dl = dl_data_module.prepare_data(processed_spectral_df)
            
            # 3. Инициализация DL-трейнера и вызов метода дообучения
            dl_trainer = DLModelTrainer(
                logger=logger, 
                experiment_name=config.EXPERIMENT_NAME, 
                sequence_length=config.DL_SEQUENCE_LENGTH, 
                lstm_units=config.DL_LSTM_UNITS, 
                epochs=config.DL_EPOCHS, 
                batch_size=config.DL_BATCH_SIZE
            )
            
            model = dl_trainer.finetune(current_mode, X_train_3D, y_train_seq, X_val_3D, y_val_seq, X_test_3D, y_test_seq, X_test_meta_dl)

            # 4. Оценка и визуализация (также, как после полного обучения)
            y_pred = dl_trainer.predict(X_test_3D)

            feature_importance = pd.Series(data=[1.0], index=['DL_MODEL'], dtype='float64')
            
            logger.warning("DL-ОЦЕНКА: Требуется дальнейшая адаптация Evaluator для 3D-данных.")
            evaluator.run(X_test_meta_dl, pd.Series(y_test_seq, index=X_test_meta_dl.index), pd.Series(y_pred, index=X_test_meta_dl.index), feature_importance)

        #=========================================================
        elif current_mode == "inference":
        #=========================================================
            logger.info("Режим: ИНФЕРЕНС (PREDICT) DL-МОДЕЛИ.")

            # 2. Подготовка DL-данных (Sequencing/Normalize) для инференса
            dl_data_module = DLDataModule(
                logger=logger,
                sequence_length=config.DL_SEQUENCE_LENGTH,
                config=config
            )
            # Для инференса нам нужен только тестовый набор, но _prepare_data вернет все сплиты.
            # Мы используем X_train.shape для архитектуры, так как она нужна для построения модели,
            # даже если мы загружаем веса.
            X_train_3D_dummy, y_train_seq_dummy, X_val_3D_dummy, y_val_seq_dummy, X_test_3D, y_test_seq, X_test_meta_dl = dl_data_module.prepare_data(processed_spectral_df)
            
            # 3. Инициализация DL-трейнера и вызов метода инференса
            dl_trainer = DLModelTrainer(
                logger=logger, 
                experiment_name=config.EXPERIMENT_NAME, 
                sequence_length=config.DL_SEQUENCE_LENGTH, 
                lstm_units=config.DL_LSTM_UNITS, 
                epochs=config.DL_EPOCHS, # Не используется в инференсе, но нужен для инициализации
                batch_size=config.DL_BATCH_SIZE # Не используется в инференсе, но нужен для инициализации
            )
            
            # Вызов метода инференса
            y_pred = dl_trainer.run_inference_mode(current_mode, X_test_3D, X_train_3D_dummy.shape)

            # 4. Оценка и визуализация (также, как после полного обучения)
            feature_importance = pd.Series(data=[1.0], index=['DL_MODEL'], dtype='float64')
            
            logger.warning("DL-ОЦЕНКА: Требуется дальнейшая адаптация Evaluator для 3D-данных.")
            evaluator.run(X_test_meta_dl, pd.Series(y_test_seq, index=X_test_meta_dl.index), pd.Series(y_pred, index=X_test_meta_dl.index), feature_importance)

    else:
        logger.error(f"Неизвестный тип модели: {config.MODEL_TYPE}")

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