# src/dl_data_preprocessor.py

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import config

class DLDataPreprocessor:
    """
    Класс для подготовки данных для DL-моделей (LSTM/CNN).
    Выполняет:
    1. Нормализацию (MinMaxScaler).
    2. Создание последовательностей (Sequencing) с помощью скользящего окна.
    """

    def __init__(self, logger: logging.Logger, sequence_length: int, config):
        """
        Инициализирует препроцессор.

        Args:
            logger (logging.Logger): Экземпляр логгера.
            sequence_length (int): Длина временной последовательности (T).
            config: Модуль конфигурации для доступа к флагам признаков.
        """
        self.logger = logger
        self.sequence_length = sequence_length
        self.config = config
### 
###         # Фильтруем input_features, чтобы оставить только ch1_
###         all_features = [col for col in input_features if 'ch1_' in col and 'centered' in col]
###         self.input_features = all_features
###         self.scaler = None # MinMaxScaler будет обучен здесь

        # Динамическое определение input_features на основе config
###         all_possible_features = [col for col in config.EXPERIMENT_CHANNELS[config.EXPERIMENT_NAME].keys() for i in range(config.N_PEAKS)]


        feature_suffixes = []
        if self.config.USE_D0_FEATURES:
            feature_suffixes.append('_amp_centered')
        if self.config.USE_D1_FEATURES:
            feature_suffixes.append('_velo')
        if self.config.USE_D2_FEATURES:
            feature_suffixes.append('_accel')

###         selected_features = []
        # Фильтруем фичи по ch1_ и выбранным производным
        # Проходим по всем колонкам в исходном датафрейме, чтобы найти соответствующие
        # Примечание: Здесь мы не имеем доступа к полному df, поэтому строим имена фичей
        # по шаблону. Это предполагает, что фичи называются 'ch1_peak_X_amp_centered' и т.д.
        # Если имена фичей другие, то эта логика потребует уточнения.

        # Более надежный способ - взять все колонки, которые могут быть фичами,
        # а затем отфильтровать их по именам.
        # Для простоты пока используем прямой шаблон
        
        # ВАЖНО! Эта логика выбора фичей должна быть более универсальной.
        # Сейчас она жестко завязана на 'ch1_'.
        # В идеале, input_features должен формироваться из X_train.columns после того,
        # как TabularModelTrainer_prepare_data определил нужные фичи, но без метаданных.
        
        # Для текущего запуска, опираясь на то, что `processed_spectral_df` уже содержит
        # `_centered`, `_velo`, `_accel` и `chX_`, мы можем построить список.
        
        # Переопределяем input_features, чтобы он включал только выбранные по конфигу
        # и только 'ch1_' по изначальному запросу, но более гибко.
        # Список всех возможных "базовых" признаков (peak_1_amp, peak_2_amp и т.д.)
        base_peak_cols = [f'peak_{i+1}' for i in range(self.config.N_PEAKS)]
        
        filtered_input_features = []
###         for bearing_idx in range(1, 5): # assuming up to 4 channels
###             for ch_idx in range(1, 3): # assuming ch1 and ch2, but only ch1 is used below
###                 if ch_idx == 1: # Only considering 'ch1_' for now based on previous code
###                     for base_col in base_peak_cols:
###                         for suffix in feature_suffixes:
###                             filtered_input_features.append(f'ch{ch_idx}_{base_col}{suffix}')

        # Формируем список признаков для 'ch1_' один раз, так как он не зависит от bearing_idx
        ch_idx_to_use = 1
        for base_col in base_peak_cols:
            for suffix in feature_suffixes:
                filtered_input_features.append(f'ch{ch_idx_to_use}_{base_col}{suffix}')
       
        self.input_features = filtered_input_features
        self.scaler = None # MinMaxScaler будет обучен здесь

    def fit(self, df: pd.DataFrame) -> None:
        """
        Обучает нормализатор (Scaler) только на обучающем наборе,
        используя только колонки, определенные как input_features.
        """
        self.logger.info("Обучение DL нормализатора (Scaler)...")

        # 1. Фильтрация и извлечение данных, используя только определенные input_features
        # Убеждаемся, что фичи существуют в DataFrame перед выбором
        features_to_fit = [col for col in self.input_features if col in df.columns]
        feature_df = df[features_to_fit].copy()

###         # Дополнительная фильтрация: удаляем колонки, которые являются константными или содержат только NaN
###         original_feature_count = len(feature_df.columns)
###         
###         # Определяем константные колонки
###         constant_cols = [col for col in feature_df.columns if feature_df[col].nunique(dropna=True) <= 1]
###         if constant_cols:
###             self.logger.warning(f"Удалены константные признаки перед масштабированием: {constant_cols}")
###             feature_df = feature_df.drop(columns=constant_cols)
###         
###         # Определяем колонки, содержащие только NaN (после удаления константных)
###         all_nan_cols = [col for col in feature_df.columns if feature_df[col].isnull().all()]
###         if all_nan_cols:
###             self.logger.warning(f"Удалены признаки, содержащие только NaN, перед масштабированием: {all_nan_cols}")
###             feature_df = feature_df.drop(columns=all_nan_cols)
### 
###         if len(feature_df.columns) < original_feature_count:
###             self.logger.info(f"Количество признаков после очистки: {len(feature_df.columns)} (удалено {original_feature_count - len(feature_df.columns)})")
###             # Обновляем self.input_features, чтобы transform_to_sequences использовал тот же набор
###             self.input_features = feature_df.columns.tolist()

        # 2. Нормализация (MinMaxScaler)
        self.scaler = MinMaxScaler()
        self.scaler.fit(feature_df)

    def transform_to_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Применяет обученный нормализатор и создает 3D-тензор.
        """
        if self.scaler is None:
            self.logger.error("Scaler не обучен! Запустите fit() перед transform_to_sequences.")
            raise RuntimeError("Scaler not fit.")
            
        self.logger.info(f"Создание DL Sequences (T={self.sequence_length}).")

        # 1. Фильтрация и извлечение данных, используя только определенные input_features
        # Убеждаемся, что фичи существуют в DataFrame перед выбором
        features_to_transform = [col for col in self.input_features if col in df.columns]
        feature_df = df[features_to_transform].copy()

        # 2. Применение нормализации (ТОЛЬКО transform)
        scaled_features = self.scaler.transform(feature_df)
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_transform, index=df.index)

        # Замена NaN (которые могут возникнуть от константных признаков после масштабирования) на 0
        nan_cols_after_scaling_transform = scaled_df.columns[scaled_df.isnull().any()].tolist()
        if nan_cols_after_scaling_transform:
            self.logger.warning(f"Признаки, ставшие NaN после масштабирования (transform): {nan_cols_after_scaling_transform}. Заменяю на 0.")
            scaled_df[nan_cols_after_scaling_transform] = scaled_df[nan_cols_after_scaling_transform].fillna(0)

        # 3. Создание последовательностей (Sliding Window)
        X_data = []
        y_data = []
        meta_indices = []

        # Группируем по подшипникам, чтобы последовательности не "перепрыгивали"
        for bearing_name in df['bearing'].unique():
            bearing_df = scaled_df[df['bearing'] == bearing_name].copy()
            bearing_rul = df[df['bearing'] == bearing_name]['RUL_hours'].copy()
            
            total_samples = len(bearing_df)
            
            if total_samples < self.sequence_length:
                self.logger.warning(f"Недостаточно данных ({total_samples} < {self.sequence_length}) для создания sequence для подшипника {bearing_name}. Пропускаем.")
                continue

            for i in range(total_samples - self.sequence_length + 1):
                # X: Берем T последовательных окон (Samples)
                X_sequence = bearing_df.iloc[i : i + self.sequence_length].values
                X_data.append(X_sequence)
                
                # Y: Берем RUL только последнего окна в последовательности (Target)
                y_target = bearing_rul.iloc[i + self.sequence_length - 1]
                y_data.append(y_target)
                
                # Метаданные: Берем метаданные последнего окна
                meta_indices.append(bearing_rul.index[i + self.sequence_length - 1])

        X_sequences = np.array(X_data)
        y_sequences = np.array(y_data)
        
        # Создаем мета-DF для соответствия RUL
        # Убеждаемся, что 'timestamp' и 'bearing' есть в исходном df
        meta_df_cols = ['timestamp', 'bearing']
        existing_meta_cols = [col for col in meta_df_cols if col in df.columns]

        meta_df = df.loc[meta_indices, existing_meta_cols].copy().reset_index(drop=True)
        meta_df['RUL_hours'] = y_sequences # Добавляем RUL для удобства
        
        self.logger.info(f"DL Sequences созданы. Форма X: {X_sequences.shape}, Форма y: {y_sequences.shape}")
        
        return X_sequences, y_sequences, meta_df

    def transform_for_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Применяет обученный скейлер и создает 3D-тензор для предсказания.
        """
        if self.scaler is None:
            self.logger.error("Scaler не обучен! Запустите fit() перед transform_for_prediction.")
            raise RuntimeError("Scaler not fit.")
            
        # 1. Фильтрация и извлечение данных, используя только определенные input_features
        features_to_transform = [col for col in self.input_features if col in df.columns]
        feature_df = df[features_to_transform].copy()

        # 2. Применение нормализации (ТОЛЬКО transform)
        scaled_features = self.scaler.transform(feature_df)
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_transform, index=df.index)
        
        # Замена NaN (которые могут возникнуть от константных признаков после масштабирования) на 0
        nan_cols_after_scaling_predict = scaled_df.columns[scaled_df.isnull().any()].tolist()
        if nan_cols_after_scaling_predict:
            self.logger.warning(f"Признаки, ставшие NaN после масштабирования (predict): {nan_cols_after_scaling_predict}. Заменяю на 0.")
            scaled_df[nan_cols_after_scaling_predict] = scaled_df[nan_cols_after_scaling_predict].fillna(0)

        # 3. Создание последовательностей (Sliding Window)
        X_data = []

        # Группируем по подшипникам, чтобы последовательности не "перепрыгивали"
        for bearing_name in df['bearing'].unique():
            bearing_df = scaled_df[df['bearing'] == bearing_name].copy()
            
            total_samples = len(bearing_df)
            
            # Если данных меньше T, мы не можем создать sequence, возвращаем пустой массив
            if total_samples < self.sequence_length:
                self.logger.warning(f"Недостаточно данных ({total_samples} < {self.sequence_length}) для создания sequence для подшипника {bearing_name}. Пропускаем.")
                ### continue

                # Возвращаем пустой массив, но с корректной размерностью
                num_features = len(self.input_features)
                return np.empty((0, self.sequence_length, num_features))
                
            for i in range(total_samples - self.sequence_length + 1):
                X_sequence = bearing_df.iloc[i : i + self.sequence_length].values
                X_data.append(X_sequence)

###         # В режиме предсказания мы берем все возможные sequences (ожидаем предсказание для каждого окна)
###         total_samples = len(scaled_df)
###         
###         # Если данных меньше T, мы не можем создать sequence, возвращаем пустой массив
###         if total_samples < self.sequence_length:
###             self.logger.warning(f"Недостаточно данных ({total_samples} < {self.sequence_length}) для создания sequence.")
###             return np.array([])
###             
###         for i in range(total_samples - self.sequence_length + 1):
###             X_sequence = scaled_df.iloc[i : i + self.sequence_length].values
###             X_data.append(X_sequence)

        X_sequences = np.array(X_data)
        
        self.logger.info(f"DL Sequences для предсказания созданы. Форма X: {X_sequences.shape}")
        
        return X_sequences