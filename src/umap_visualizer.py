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

class UMAPVisualizer:
    """
    Выполняет снижение размерности с помощью UMAP и создает
    анимированную визуализацию эволюции состояния подшипников.
    """

    def __init__(self, output_path: pathlib.Path, logger: logging.Logger, sample_fraction: float, animation_frequency: str, animation_interval: int):
        """
        Инициализирует визуализатор.

        Args:
            output_path (pathlib.Path): Путь для сохранения итогового GIF-файла.
            logger (logging.Logger): Экземпляр логгера.
            animation_frequency (str): Единица измерения частоты кадров ('D', 'H', 'T').
            animation_interval (int): Интервал между кадрами.
            sample_fraction (float): Доля данных для использования в анализе.
        """
        self.output_path = output_path
        self.logger = logger
        self.sample_fraction = sample_fraction
        self.animation_frequency = animation_frequency
        self.animation_interval = animation_interval

    def run(self, spectral_df: pd.DataFrame) -> pd.DataFrame:
        """
        Главный метод. Запускает предобработку и (в будущем) визуализацию.
        """
        self.logger.info("Запуск пайплайна UMAP визуализации...")
        
        processed_df = self._preprocess_data(spectral_df)

        # Выполняем UMAP преобразование
        umap_df = self._calculate_umap_embeddings(processed_df)
        
        # Создаем и сохраняем статичный график
        self._plot_static_umap(umap_df)

        # Создаем две анимированные визуализации
        self._create_animation_by_bearing(umap_df)
        self._create_animation_by_time(umap_df)
        self._create_migrational_animation(umap_df)

        self.logger.info("Пайплайн UMAP визуализации завершен.")
        return umap_df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Готовит датасет со спектральными признаками для UMAP.
        - Удаляет неинформативный первый пик.
        - Заполняет пропуски нулями.
        """
        self.logger.info("Подготовка данных: удаление Пика 1 и заполнение пропусков.")
        processed_df = df.copy()

        # 1. Удаляем "белый шум" - самый сильный, но неинформативный пик
        cols_to_drop = ['peak_1_freq', 'peak_1_amp']
        if all(col in processed_df.columns for col in cols_to_drop):
            processed_df.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(f"Удалены колонки: {cols_to_drop}")
        else:
            self.logger.warning("Колонки для удаления (peak_1_*) не найдены. Пропускаем шаг.")

        # 2. Заполняем все пропуски в оставшихся пиках нулями,
        # так как NaN означает отсутствие значимого пика.
        peak_cols = [col for col in processed_df.columns if 'peak' in col]
        processed_df[peak_cols] = processed_df[peak_cols].fillna(0)
        self.logger.info("Пропуски в данных о пиках (NaN) успешно заменены на 0.")

        return processed_df

    def _calculate_umap_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет UMAP преобразование для снижения размерности.
        """
        self.logger.info(f"Применяем UMAP к последним {self.sample_fraction*100}% данных...")
        
        # Берем только числовые колонки с признаками для UMAP
        feature_cols = [col for col in df.columns if 'peak' in col]
        
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
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(scaled_features)
        
        # Собираем результат в новый DataFrame
        umap_result_df = pd.DataFrame(embedding, columns=['umap_x', 'umap_y'])
        # Добавляем обратно метаданные для раскраски графика
        umap_result_df['bearing'] = umap_input_df['bearing'].values
        umap_result_df['timestamp'] = pd.to_datetime(umap_input_df['timestamp'], format='%Y.%m.%d.%H.%M.%S').values
        self.logger.info("UMAP преобразование успешно завершено.")
        return umap_result_df

    def _plot_static_umap(self, df: pd.DataFrame):
        """
        Создает и сохраняет статичный 2D-график UMAP-эмбеддингов.
        """
        self.logger.info("Создание статичного UMAP-графика...")
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
        
        plt.title('UMAP проекция спектральных признаков', fontsize=16)
        plt.xlabel('UMAP компонента 1', fontsize=12)
        plt.ylabel('UMAP компонента 2', fontsize=12)
        plt.grid(True)
        
        save_path = self.output_path.with_suffix('.png') # Сохраняем как PNG
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

    def _create_animation_by_bearing(self, df: pd.DataFrame):
        """Создает GIF-анимацию, где точки раскрашены по номеру подшипника."""
        
        save_path = self.output_path.with_name(f"{self.output_path.stem}_by_bearing.gif")
        self.logger.info(f"Создание GIF-анимации по подшипникам... Это может занять несколько минут. Результат будет в {save_path}")

        # Подготовка данных для анимации
        frame_timestamps = self._generate_frame_timestamps(df)

        ### # Добавляем паузу в конце, повторяя последний кадр
        ### fps = 5  # Кадров в секунду
        ### pause_sec = 5 # Длительность паузы в секундах
        ### last_day = unique_days[-1]
        ### unique_days.extend([last_day] * (fps * pause_sec))

        fig, ax = plt.subplots(figsize=(12, 10))

        # Находим глобальные границы для осей, чтобы график не "прыгал"
        x_min, x_max = df['umap_x'].min() - 1, df['umap_x'].max() + 1
        y_min, y_max = df['umap_y'].min() - 1, df['umap_y'].max() + 1

        def update(frame_timestamp):
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
            ax.set_title(f'Эволюция кластеров по подшипникам\nДата: {frame_timestamp.strftime("%Y-%m-%d %H:%M")}', fontsize=16)
            ax.set_xlabel('UMAP компонента 1', fontsize=12)
            ax.set_ylabel('UMAP компонента 2', fontsize=12)
            ax.grid(True)
            return artists,

        # Создаем анимацию
        ani = FuncAnimation(fig, update, frames=frame_timestamps, repeat=False)
        ani.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
        self.logger.info("Анимация по подшипникам успешно создана.")

    def _create_animation_by_time(self, df: pd.DataFrame):
        """Создает GIF-анимацию, где точки раскрашены по времени (RUL)."""

        save_path = self.output_path.with_name(f"{self.output_path.stem}_by_time.gif")
        self.logger.info(f"Создание GIF-анимации по времени... Это может занять несколько минут. Результат будет в {save_path}")

        ### df['day'] = df['timestamp'].dt.date
        ### unique_days = sorted([d for d in df['day'].unique() if pd.notna(d)])
        frame_timestamps = self._generate_frame_timestamps(df)

        ### # Добавляем паузу в конце
        ### fps = 5
        ### pause_sec = 5
        ### last_day = unique_days[-1]
        ### unique_days.extend([last_day] * (fps * pause_sec))

        fig, ax = plt.subplots(figsize=(12, 10))

        x_min, x_max = df['umap_x'].min() - 1, df['umap_x'].max() + 1
        y_min, y_max = df['umap_y'].min() - 1, df['umap_y'].max() + 1

        # Нормализуем временные метки для градиентной раскраски
        norm = Normalize(df['timestamp'].min().timestamp(), df['timestamp'].max().timestamp())
        cmap = plt.get_cmap('plasma')


        def update(frame_timestamp):
            ax.clear()
            data_so_far = df[df['timestamp'] <= frame_timestamp]
            
            points = ax.scatter(
                data_so_far['umap_x'], data_so_far['umap_y'],
                c=[cmap(norm(ts.timestamp())) for ts in data_so_far['timestamp']],
                alpha=0.6, s=10
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Эволюция процесса деградации по времени\nДата: {frame_timestamp.strftime("%Y-%m-%d %H:%M")}', fontsize=16)
            ax.set_xlabel('UMAP компонента 1', fontsize=12)
            ax.set_ylabel('UMAP компонента 2', fontsize=12)
            ax.grid(True)
            return points,
        
        ani = FuncAnimation(fig, update, frames=frame_timestamps, repeat=False)
        ani.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
        self.logger.info("Анимация по времени успешно создана.")

    def _create_migrational_animation(self, df: pd.DataFrame):
        """Создает GIF, показывающий 'миграцию' кластеров во времени."""
        
        save_path = self.output_path.with_name(f"{self.output_path.stem}_migration.gif")
        self.logger.info(f"Создание миграционной GIF-анимации... Результат будет в {save_path}")

        df['day'] = df['timestamp'].dt.date
        unique_days = sorted([d for d in df['day'].unique() if pd.notna(d)])
        frame_timestamps = self._generate_frame_timestamps(df)

        fig, ax = plt.subplots(figsize=(12, 10))
        x_min, x_max = df['umap_x'].min() - 1, df['umap_x'].max() + 1
        y_min, y_max = df['umap_y'].min() - 1, df['umap_y'].max() + 1
        
        # Окно в днях для отображения
        window_days = 3

        def update(frame_idx):
            ax.clear()
            
            # Определяем текущий день и границы окна
            current_day = unique_days[frame_idx]
            start_day = current_day - pd.to_timedelta(window_days, unit='d')
            
            data_in_window = df[(df['day'] > start_day) & (df['day'] <= current_day)]
            
            # Создаем градиент "свежести"
            norm = Normalize(start_day.toordinal(), current_day.toordinal())
            cmap = plt.get_cmap('hot') # "Горячая" палитра: от темного к ярко-желтому

            points = ax.scatter(
                data_in_window['umap_x'], data_in_window['umap_y'],
                c=[cmap(norm(d.toordinal())) for d in data_in_window['day']],
                alpha=0.7, s=15
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Миграция кластеров (окно: {window_days} дня)\nДата: {current_day.strftime("%Y-%m-%d")}', fontsize=16)
            ax.set_xlabel('UMAP компонента 1', fontsize=12)
            ax.set_ylabel('UMAP компонента 2', fontsize=12)
            ax.grid(True)
            return points,

        ani = FuncAnimation(fig, update, frames=range(window_days, len(frame_timestamps)), repeat=False)
        ani = FuncAnimation(fig, update, frames=range(window_days, len(unique_days)), repeat=False)
        ani.save(save_path, writer='pillow', fps=5)
        plt.close(fig)
        self.logger.info("Миграционная анимация успешно создана.")