# src/umap_visualizer.py

import matplotlib
matplotlib.use('Agg') # Устанавливаем "неинтерактивный" бэкенд ПЕРЕД импортом pyplot

import logging
import pathlib
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import matplotlib.patches as patches

class UMAPVisualizer:
    """
    Выполняет снижение размерности с помощью UMAP и создает
    анимированную визуализацию эволюции состояния подшипников.
    """

    def __init__(self, output_path: pathlib.Path, logger: logging.Logger, sample_fraction: float, animation_frequency: str, animation_interval: int, experiment_name: str, migration_window_days: int):
        """
        Инициализирует визуализатор.

        Args:
            output_path (pathlib.Path): Путь для сохранения итогового GIF-файла.
            logger (logging.Logger): Экземпляр логгера.
            animation_frequency (str): Единица измерения частоты кадров ('D', 'H', 'T').
            animation_interval (int): Интервал между кадрами.
            sample_fraction (float): Доля данных для использования в анализе.
            experiment_name (str): Имя текущего эксперимента.
            migration_window_days (int): Длина окна для миграционной гифки.
        """
        self.output_path = output_path
        self.logger = logger
        self.sample_fraction = sample_fraction
        self.animation_frequency = animation_frequency
        self.animation_interval = animation_interval
        self.experiment_name = experiment_name
        self.migration_window_days = migration_window_days

    def run(self, spectral_df: pd.DataFrame, derivative_level: str) -> pd.DataFrame:
        """
        Главный метод. Запускает предобработку и (в будущем) визуализацию.
        """
        self.logger.info(f"Запуск UMAP визуализации для уровня: {derivative_level}")
        processed_df = self._preprocess_data(spectral_df, derivative_level)
        
        # Выполняем UMAP преобразование
        umap_df = self._calculate_umap_embeddings(processed_df)
        
        # Создаем и сохраняем статичный 
        self._plot_static_umap(umap_df, derivative_level)

        # Создаем две анимированные визуализации
        self._create_animation_by_bearing(umap_df, derivative_level)
        self._create_animation_by_time(umap_df, derivative_level)
        self._create_migrational_animation(umap_df, derivative_level)

        self.logger.info(f"Пайплайн UMAP визуализации завершен для уровня: {derivative_level}")
        return umap_df

    def _preprocess_data(self, df: pd.DataFrame, derivative_level: str) -> pd.DataFrame:
        """
        Готовит датасет для UMAP, выбирая нужный набор признаков.
        - 'd0': Исходные пики (без производных)
        - 'd1': Только скорости (velo)
        - 'd2': Только ускорения (accel)
        """
        self.logger.info("Подготовка данных: удаление Пика 1 и заполнение пропусков.")

        # Выбираем, какие признаки оставить
        if derivative_level == 'd0':
            # Исходные: есть 'peak', но нет 'velo' и 'accel'
            feature_cols = [c for c in df.columns if 'peak' in c and 'velo' not in c and 'accel' not in c]
        elif derivative_level == 'd1':
            feature_cols = [c for c in df.columns if 'velo' in c]
        elif derivative_level == 'd2':
            feature_cols = [c for c in df.columns if 'accel' in c]
        else:
            raise ValueError("Неверный уровень производной. Используйте 'd0', 'd1' или 'd2'.")

        # Оставляем только нужные признаки и метаданные
        meta_cols = ['timestamp', 'window_id', 'bearing']
        processed_df = df[meta_cols + feature_cols].copy()

        # 2. Заполняем все пропуски в оставшихся пиках нулями,
        # так как NaN означает отсутствие значимого пика.

        peak_cols = [col for col in processed_df.columns if 'peak' in col or 'velo' in col or 'accel' in col]
        processed_df[peak_cols] = processed_df[peak_cols].fillna(0)
        self.logger.info("Пропуски в данных о признаках (NaN) успешно заменены на 0.")

        return processed_df

    def _calculate_umap_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет UMAP преобразование для снижения размерности.
        """
        self.logger.info(f"Применяем UMAP к последним {self.sample_fraction*100}% данных...")
        
        # Берем только числовые колонки с признаками для UMAP
        feature_cols = [col for col in df.columns if 'peak' in col or 'velo' in col or 'accel' in col]

        # Если задана выборка, используем ее
        if self.sample_fraction < 1.0:
            num_rows = int(len(df) * self.sample_fraction)
            umap_input_df = df.tail(num_rows)
        else:
            umap_input_df = df
            
        features = umap_input_df[feature_cols]
        
        # Стандартизация данных - важный шаг перед UMAP
        self.logger.info("Стандартизация признаков...")
        scaled_features = StandardScaler().fit_transform(features)
        
        # Инициализация и обучение UMAP
        self.logger.info("Обучение UMAP модели...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, verbose=True)
        embedding = reducer.fit_transform(scaled_features)
        
        # Собираем результат в новый DataFrame
        umap_result_df = pd.DataFrame(embedding, columns=['umap_x', 'umap_y'])
        # Добавляем обратно метаданные для раскраски графика
        umap_result_df['bearing'] = umap_input_df['bearing'].values
        umap_result_df['timestamp'] = pd.to_datetime(umap_input_df['timestamp'], format='%Y.%m.%d.%H.%M.%S').values
        self.logger.info("UMAP преобразование успешно завершено.")
        return umap_result_df

    def _plot_static_umap(self, df: pd.DataFrame, derivative_level: str):
        """
        Создает и сохраняет статичный 2D-график UMAP-эмбеддингов.
        """
        self.logger.info(f"Создание статичного UMAP-графика для '{derivative_level}'...")
        plt.figure(figsize=(12, 10))
        
        # Используем seaborn для красивой раскраски по подшипникам
        sns.scatterplot(
            x='umap_x',
            y='umap_y',
            hue='bearing', # Раскрашиваем точки в зависимости от подшипника
            palette=sns.color_palette("hsv", n_colors=df['bearing'].nunique()),
            data=df,
            legend="full",
            alpha=0.5, # Делаем точки полупрозрачными
            s=10 # Уменьшаем размер точек
        )
        
        plt.title(f'UMAP проекция ({derivative_level})', fontsize=16)
        plt.xlabel('UMAP компонента 1', fontsize=12)
        plt.ylabel('UMAP компонента 2', fontsize=12)
        plt.grid(True)
        
        save_path = self.output_path.with_name(f"{self.experiment_name}_evo_static_{derivative_level}.png") # Сохраняем как PNG
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Статичный UMAP-график сохранен: {save_path}")

    def _generate_frame_timestamps(self, df: pd.DataFrame) -> list:
        """Создает список временных меток для кадров анимации."""
        freq_str = f"{self.animation_interval}{self.animation_frequency}"
        self.logger.info(f"Генерация кадров с частотой: 1 кадр каждые {self.animation_interval} {self.animation_frequency}")
        
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        
        frame_timestamps = pd.date_range(start=min_ts, end=max_ts, freq=freq_str).tolist()
        
        # Добавляем паузу в конце
        fps = 5
        pause_sec = 5
        last_ts = frame_timestamps[-1]
        frame_timestamps.extend([last_ts] * (fps * pause_sec))
        
        return frame_timestamps

    def _draw_progress_bars(self, ax, current_frame_idx, total_real_frames, pause_start_idx, total_all_frames):
        """Рисует прогресс-бар анимации и паузы внизу графика."""
        # Прогресс основной анимации (синий)
        progress = min(1.0, (current_frame_idx + 1) / total_real_frames)
        ax.add_patch(patches.Rectangle((0, -0.1), 1, 0.02, facecolor='#E0E0E0', transform=ax.transAxes, clip_on=False, zorder=5))
        ax.add_patch(patches.Rectangle((0, -0.1), progress, 0.02, facecolor='#007ACC', transform=ax.transAxes, clip_on=False, zorder=5))
        
        # Прогресс паузы (оранжевый) - работает как обратный отсчет
        pause_progress = 1.0 # По умолчанию бар полный
        if current_frame_idx >= pause_start_idx:
            # Считаем, сколько кадров осталось до конца паузы
            total_pause_frames = total_all_frames - pause_start_idx
            remaining_pause_frames = total_all_frames - current_frame_idx - 1
            if total_pause_frames > 0:
                pause_progress = remaining_pause_frames / total_pause_frames

        ax.add_patch(patches.Rectangle((0, -0.13), 1, 0.02, facecolor='#E0E0E0', transform=ax.transAxes, clip_on=False))
        ax.add_patch(patches.Rectangle((0, -0.13), pause_progress, 0.02, facecolor='#FF4500', transform=ax.transAxes, clip_on=False))

    def _create_animation_by_bearing(self, df: pd.DataFrame, derivative_level: str):
        """Создает GIF-анимацию, где точки раскрашены по номеру подшипника."""
        
        save_path = self.output_path.with_name(f"{self.output_path.stem}_by_bearing_{derivative_level}.gif")
        self.logger.info(f"Создание GIF-анимации по подшипникам для '{derivative_level}... Это может занять несколько минут. Результат будет в {save_path}")
        
        # Подготовка данных для анимации
        frame_timestamps = self._generate_frame_timestamps(df)

        # Определяем точку начала паузы
        total_real_frames = len(set(frame_timestamps))
        pause_start_index = total_real_frames -1
        
        fig, ax = plt.subplots(figsize=(12, 10))

        # Находим глобальные границы для осей, чтобы график не "прыгал"
        x_min, x_max = df['umap_x'].min() - 1, df['umap_x'].max() + 1
        y_min, y_max = df['umap_y'].min() - 1, df['umap_y'].max() + 1

        def update(frame_idx):
            frame_timestamp = frame_timestamps[frame_idx]
            ax.clear()
            
            data_so_far = df[df['timestamp'] <= frame_timestamp]

            artists = sns.scatterplot(
            # sns.scatterplot(
                x='umap_x', y='umap_y', hue='bearing',
                palette=sns.color_palette("hsv", n_colors=df['bearing'].nunique()),
                data=data_so_far, legend="full", alpha=0.6, s=10, ax=ax
            )
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Эволюция кластеров по подшипникам ({derivative_level})\n(IMS Dataset: {self.experiment_name}) | Дата: {frame_timestamp.strftime("%Y-%m-%d %H:%M")}', fontsize=16)
            ax.set_xlabel('UMAP компонента 1', fontsize=12)
            ax.set_ylabel('UMAP компонента 2', fontsize=12)
            ax.grid(True)
            self._draw_progress_bars(ax, frame_idx, total_real_frames, pause_start_index, len(frame_timestamps))
            return artists,

        # Создаем анимацию
        ani = FuncAnimation(fig, update, frames=len(frame_timestamps), repeat=False)
        ani.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
        self.logger.info("Анимация по подшипникам успешно создана.")

    def _create_animation_by_time(self, df: pd.DataFrame, derivative_level: str):
        """Создает GIF-анимацию, где точки раскрашены по времени (RUL)."""

        save_path = self.output_path.with_name(f"{self.output_path.stem}_by_time_{derivative_level}.gif")
        self.logger.info(f"Создание GIF-анимации по времени для '{derivative_level}... Это может занять несколько минут. Результат будет в {save_path}")
        
        ### df['day'] = df['timestamp'].dt.date
        ### unique_days = sorted([d for d in df['day'].unique() if pd.notna(d)])
        frame_timestamps = self._generate_frame_timestamps(df)

        # Определяем точку начала паузы для прогресс-бара
        total_real_frames = len(set(frame_timestamps))
        pause_start_index = total_real_frames - 1
        
        fig, ax = plt.subplots(figsize=(12, 10))

        x_min, x_max = df['umap_x'].min() - 1, df['umap_x'].max() + 1
        y_min, y_max = df['umap_y'].min() - 1, df['umap_y'].max() + 1

        # Нормализуем временные метки для градиентной раскраски
        norm = Normalize(df['timestamp'].min().timestamp(), df['timestamp'].max().timestamp())
        cmap = plt.get_cmap('plasma')

        def update(frame_idx):
            frame_timestamp = frame_timestamps[frame_idx]
            ax.clear()
            data_so_far = df[df['timestamp'] <= frame_timestamp]
            
            points = ax.scatter(
                data_so_far['umap_x'], data_so_far['umap_y'],
                c=[cmap(norm(ts.timestamp())) for ts in data_so_far['timestamp']],
                alpha=0.6, s=10
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Эволюция процесса деградации по времени ({derivative_level})\n(IMS Dataset: {self.experiment_name}) | Дата: {frame_timestamp.strftime("%Y-%m-%d %H:%M")}', fontsize=16)
            ax.set_xlabel('UMAP компонента 1', fontsize=12)
            ax.set_ylabel('UMAP компонента 2', fontsize=12)
            ax.grid(True)
            self._draw_progress_bars(ax, frame_idx, total_real_frames, pause_start_index, len(frame_timestamps))
            return points,

        ani = FuncAnimation(fig, update, frames=len(frame_timestamps), repeat=False)
        ani.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
        self.logger.info("Анимация по времени успешно создана.")

    def _create_migrational_animation(self, df: pd.DataFrame, derivative_level: str):
        """Создает GIF, показывающий 'миграцию' кластеров во времени."""
        
        save_path = self.output_path.with_name(f"{self.output_path.stem}_migration_{derivative_level}.gif")
        self.logger.info(f"Создание миграционной GIF-анимации для '{derivative_level}... Результат будет в {save_path}")

        frame_timestamps = self._generate_frame_timestamps(df)

        # Определяем точку начала паузы для прогресс-бара
        total_real_frames = len(set(frame_timestamps))
        pause_start_index = total_real_frames - 1

        fig, ax = plt.subplots(figsize=(12, 10))
        x_min, x_max = df['umap_x'].min() - 1, df['umap_x'].max() + 1
        y_min, y_max = df['umap_y'].min() - 1, df['umap_y'].max() + 1
        
        # Окно в днях для отображения
        window_days = self.migration_window_days

        def update(frame_idx):
            ax.clear()
            
            # Определяем текущий день и границы окна
            current_ts = frame_timestamps[frame_idx]
            start_ts = current_ts - pd.to_timedelta(window_days, unit='d')
            
            data_in_window = df[(df['timestamp'] > start_ts) & (df['timestamp'] <= current_ts)]
                        
            # Создаем градиент "свежести"
            norm = Normalize(start_ts.timestamp(), current_ts.timestamp())
            cmap = plt.get_cmap('hot') # "Горячая" палитра: от темного к ярко-желтому

            points = ax.scatter(
                data_in_window['umap_x'], data_in_window['umap_y'],
                c=[cmap(norm(ts.timestamp())) for ts in data_in_window['timestamp']],
                alpha=0.7, s=15
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Миграция кластеров ({derivative_level}) [окно: {window_days} дня]\n(IMS Dataset: {self.experiment_name}) | Дата: {current_ts.strftime("%Y-%m-%d %H:%M")}', fontsize=16)
            ax.set_xlabel('UMAP компонента 1', fontsize=12)
            ax.set_ylabel('UMAP компонента 2', fontsize=12)
            ax.grid(True)
            self._draw_progress_bars(ax, frame_idx, total_real_frames, pause_start_index, len(frame_timestamps))
            return points,

        # Запускаем анимацию со сдвигом, чтобы окно было полным
        start_frame = len(pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].min() + pd.to_timedelta(window_days, unit='d'), freq=f"{self.animation_interval}{self.animation_frequency}"))
        ani = FuncAnimation(fig, update, frames=range(start_frame, len(frame_timestamps)), repeat=False)
        ani.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
        self.logger.info("Миграционная анимация успешно создана.")