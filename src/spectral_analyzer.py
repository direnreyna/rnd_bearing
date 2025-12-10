# src/spectral_analyzer.py

import logging
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

class SpectralAnalyzer:
    """
    Извлекает спектральные признаки из сырых временных рядов.
    1. Нарезает сырой сигнал на перекрывающиеся окна.
    2. Для каждого окна применяет FFT.
    3. Находит N самых сильных частотных пиков (амплитуда + частота).
    """

    def __init__(self, raw_data_path: pathlib.Path, spectral_filepath: pathlib.Path, logger: logging.Logger, config, window_size: int, step: int, n_peaks: int, sampling_rate: int):
        self.raw_data_path = raw_data_path
        self.spectral_filepath = spectral_filepath
        self.logger = logger
        self.window_size = window_size
        self.step = step
        self.n_peaks = n_peaks
        self.sampling_rate = sampling_rate
        self.config = config

    def run(self) -> pd.DataFrame:
        """
        Главный метод. Управляет кэшированием и запуском процесса.
        """
        self.logger.info("Запуск извлечения спектральных признаков для UMAP визуализации")
        if self.spectral_filepath.exists():
            self.logger.info(f"Загружаем спектральные признаки из кэша: {self.spectral_filepath}")
            spectral_df = pd.read_parquet(self.spectral_filepath)
        else:
            self.logger.info("Кэш спектральных признаков не найден. Запускаем полный анализ...")
            spectral_df = self._create_spectral_dataset()

        self.logger.info(f"Датасет спектральных признаков готов. Форма: {spectral_df.shape}")
        return spectral_df
            
###     def _create_spectral_dataset(self) -> pd.DataFrame:
###         """
###         Итерируется по всем сырым файлам и собирает датасет спектральных признаков.
###         """
###         file_list = sorted([f for f in self.raw_data_path.iterdir() if f.is_file()])
###         all_features = []
### 
###         for file_path in tqdm(file_list, desc="Анализ спектров"):
###             timestamp_str = file_path.name
###             raw_df = pd.read_csv(file_path, sep='\t', header=None)
###             
###             # Обрабатываем каждый подшипник (каждую колонку)
###             for col_idx, col_name in enumerate(raw_df.columns):
###                 series = raw_df[col_name]
###                 bearing_name = f'bearing_{col_idx + 1}'
###                 windows = self._create_windows(series.to_numpy())
###                 
###                 for window_idx, window in enumerate(windows):
###                     features = self._extract_fft_features(window)
###                     features['timestamp'] = timestamp_str
###                     features['window_id'] = f'{timestamp_str}_{bearing_name}_{window_idx}'
###                     features['bearing'] = bearing_name
###                     all_features.append(features)
###         
###         spectral_df = pd.DataFrame(all_features)
###         self.logger.info(f"Сохраняем датасет спектральных признаков: {self.spectral_filepath}")

    def _create_spectral_dataset(self) -> pd.DataFrame:
        """
        Итерируется по всем сырым файлам и собирает датасет спектральных признаков.
        """
        file_list = sorted([f for f in self.raw_data_path.iterdir() if f.is_file()])
        all_features = []
        
        # Получаем карту каналов для текущего эксперимента
        experiment_name = self.raw_data_path.name
        channel_map = self.config.EXPERIMENT_CHANNELS[experiment_name]

        for file_path in tqdm(file_list, desc="Анализ спектров"):
            timestamp_str = file_path.name
            raw_df = pd.read_csv(file_path, sep='\t', header=None)
            
            windows_by_timestamp = {}

            # Итерируемся по подшипникам, а не по колонкам
            for bearing_name, channel_indices in channel_map.items():
                
                # Собираем данные со всех каналов для одного подшипника
                # В 99% случаев это будет 1 или 2 канала
                bearing_signals = [raw_df[idx].to_numpy() for idx in channel_indices]
                
                # Нарезаем на окна. Убеждаемся, что все каналы имеют одинаковое число окон.
                # (Берем за основу первый канал)
                num_windows = (len(bearing_signals[0]) - self.window_size) // self.step + 1

                for i in range(num_windows):
                    window_id = f'{timestamp_str}_{bearing_name}_{i}'
                    window_features = {'timestamp': timestamp_str, 'bearing': bearing_name, 'window_id': window_id}

                    # Для каждого канала извлекаем свой набор пиков
                    for ch_idx, signal in enumerate(bearing_signals):
                        window = signal[i*self.step : i*self.step + self.window_size]
                        fft_features = self._extract_fft_features(window)
                        
                        # Добавляем префикс канала (ch1, ch2) к именам признаков
                        for key, value in fft_features.items():
                            window_features[f'ch{ch_idx+1}_{key}'] = value
                    
                    all_features.append(window_features)
        
        spectral_df = pd.DataFrame(all_features)
        self.logger.info(f"Сохраняем датасет спектральных признаков: {self.spectral_filepath}")
        spectral_df.to_parquet(self.spectral_filepath)
        return spectral_df

    def _create_windows(self, data: np.ndarray) -> list[np.ndarray]:
        """Нарезает одномерный массив на перекрывающиеся окна."""
        num_windows = (len(data) - self.window_size) // self.step + 1
        windows = [data[i*self.step : i*self.step + self.window_size] for i in range(num_windows)]
        return windows

    def _extract_fft_features(self, window: np.ndarray) -> dict:
        """Применяет FFT и извлекает N самых сильных пиков."""
        n = len(window)
        # Применяем FFT
        yf = np.fft.fft(window)
        xf = np.fft.fftfreq(n, 1 / self.sampling_rate)

        # Берем только положительную часть спектра
        yf_abs = 2.0/n * np.abs(yf[:n//2])
        xf_pos = xf[:n//2]
        
        # Находим пики
        peaks_idx, _ = find_peaks(yf_abs, height=0.01) # height - минимальная амплитуда пика
        
        # Сортируем пики по амплитуде и берем топ N
        sorted_peak_indices = sorted(peaks_idx, key=lambda i: yf_abs[i], reverse=True)[:self.n_peaks]

        features = {}
        for i, peak_idx in enumerate(sorted_peak_indices):
            features[f'peak_{i+1}_freq'] = xf_pos[peak_idx]
            features[f'peak_{i+1}_amp'] = yf_abs[peak_idx]
            
        return features