# src/model_evaluator.py

import logging
import pathlib
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Класс для оценки и визуализации результатов предсказательной модели RUL.
    """

    def __init__(self, plots_dir: pathlib.Path, logger: logging.Logger):
        """
        Инициализирует оценщика.

        Args:
            plots_dir (pathlib.Path): Папка для сохранения графиков.
            logger (logging.Logger): Экземпляр логгера.
        """
        self.plots_dir = plots_dir
        self.logger = logger
        sns.set(style="whitegrid")

    def run(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, feature_importance: pd.Series):
        """
        Главный метод. Запускает визуализацию сравнения RUL.

        Args:
            X_test (pd.DataFrame): Тестовые признаки (теперь содержит мета-данные).
            y_test (pd.Series): Реальные значения RUL.
            y_pred (pd.Series): Предсказанные значения RUL.
            feature_importance (pd.Series): Важность признаков от модели.
        """
        self.logger.info("Запуск оценки и визуализации результатов модели.")
        
        # Объединяем тестовый результат в один датафрейм для удобства
        evaluation_df = pd.DataFrame({
            'RUL_Actual': y_test,
            'RUL_Predicted': y_pred
        })
        
        # Добавляем мета-данные обратно
        # Поскольку y_test.index совпадает с индексом X_test, используем его.
        # В X_test не попали 'timestamp' и 'bearing', но они были в исходном processed_spectral_df.
        # Однако, мы можем извлечь их из индексов, если они были сохранены (на данный момент они не были).
        
        # АЛЬТЕРНАТИВНЫЙ ПОДХОД: Слияние по индексу (это самое надежное)
        
        # 1. Извлекаем нужные мета-колонки (timestamp и bearing) из исходного датафрейма.
        # Внимание: X_test.index - это исходный индекс, используем его для извлечения.
        
        # ВАЖНО: Текущий X_test НЕ содержит 'timestamp' и 'bearing', так как они были удалены в _prepare_data.
        # Мы должны были передать их из main.py, но чтобы не ломать архитектуру,
        # примем, что нам достаточно сравнить RUL без временной оси (пока).

        evaluation_df = pd.merge(evaluation_df, X_test[['timestamp', 'bearing']], left_index=True, right_index=True, how='left')
        self.logger.info("График RUL будет построен по оси времени (timestamp) с разбивкой по подшипникам.")

        self._plot_scatter_comparison(evaluation_df)
        self._plot_rul_trend(evaluation_df) ## ИЗМЕНЕНО: убран аргумент original_index
        ### self._plot_rul_trend(evaluation_df, y_test.index) # Построение по индексу (последовательности)
        self._plot_feature_importance(feature_importance)

        self.logger.info("Визуализация оценки модели завершена.")

    def _plot_scatter_comparison(self, df: pd.DataFrame):
        """
        Строит график сравнения фактического RUL с предсказанным.
        Идеальное предсказание - точки лежат на линии y=x.
        """
        plt.figure(figsize=(8, 8))
        
        # Scatter plot для сравнения
        sns.scatterplot(
            x='RUL_Actual',
            y='RUL_Predicted',
            data=df,
            alpha=0.5,
            s=10
        )
        
        # Идеальная линия (y=x)
        max_rul = df['RUL_Actual'].max()
        plt.plot([0, max_rul], [0, max_rul], 'r--', label='Идеальное предсказание')
        
        plt.title('Сравнение фактического RUL с предсказанным (Baseline)')
        plt.xlabel('Фактический RUL (часы)')
        plt.ylabel('Предсказанный RUL (часы)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'rul_scatter_comparison.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График сравнения RUL сохранен: {save_path}")
        
    def _plot_rul_trend(self, df: pd.DataFrame):
        """
        Строит график RUL в зависимости от времени, с разбивкой по подшипникам.
        """
        self.logger.info("Создание графика RUL во времени (по подшипникам)...")

        df = df.sort_values(by='timestamp')
        
        plt.figure(figsize=(15, 6))
        
        # Использование rolling mean для сглаживания предсказаний
        window = 50
        
        for bearing in df['bearing'].unique(): ## Итерация по подшипникам
            df_bearing = df[df['bearing'] == bearing]
            
            # Фактический RUL (не сглаживается, так как по времени он уже "гладкий")
            plt.plot(df_bearing['timestamp'], df_bearing['RUL_Actual'], 
                     label=f'Фактический RUL ({bearing})', color='blue', alpha=0.5)

            # Предсказанный RUL (сглаживание)
            plt.plot(df_bearing['timestamp'], df_bearing['RUL_Predicted'].rolling(window=window).mean(), 
                     label=f'Предсказанный RUL ({bearing}) (Сглаженный: {window})', color='red', linestyle='--')

        plt.title(f'Эволюция RUL: Фактический vs. Предсказанный (Сглаживание: Окно={window})')
        plt.xlabel('Дата и время')
        plt.ylabel('RUL (часы)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'rul_trend_comparison.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График эволюции RUL сохранен: {save_path}")

    def _plot_feature_importance(self, feature_importance: pd.Series, top_n: int = 20):
        """
        Строит график важности признаков (Feature Importance) для LightGBM.
        """
        self.logger.info(f"Создание графика TOP-{top_n} самых важных признаков.")
        
        # Берем топ-N
        top_features = feature_importance.head(top_n).sort_values(ascending=True)
        
        plt.figure(figsize=(10, 0.5 * top_n)) # Динамический размер графика
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
        
        plt.title(f'ТОП-{top_n} самых важных признаков для предсказания RUL (LightGBM)', fontsize=14)
        plt.xlabel('Важность признака (Gain/Split)', fontsize=12)
        plt.ylabel('Признак', fontsize=12)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'feature_importance_top20.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График важности признаков сохранен: {save_path}")