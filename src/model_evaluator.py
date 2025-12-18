# src/model_evaluator.py

import logging
import pathlib

from typing import Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Класс для оценки и визуализации результатов предсказательной модели RUL.
    """

    def __init__(self, plots_dir: pathlib.Path, logger: logging.Logger, model_type: str):
        """
        Инициализирует оценщика.

        Args:
            plots_dir (pathlib.Path): Папка для сохранения графиков.
            logger (logging.Logger): Экземпляр логгера.
            model_type (str): Тип используемой модели (DL, LGBM, CATB) для формирования имен файлов.
        """
        self.plots_dir = plots_dir
        self.logger = logger
        self.model_type = model_type
        sns.set(style="whitegrid")

    def run(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, feature_importance: pd.Series, dl_history: Any = None):
        """
        Главный метод. Запускает визуализацию сравнения RUL.

        Args:
            X_test (pd.DataFrame): Тестовые признаки (теперь содержит мета-данные).
            y_test (pd.Series): Реальные значения RUL.
            y_pred (pd.Series): Предсказанные значения RUL.
            feature_importance (pd.Series): Важность признаков от модели.
            dl_history (Any): Объект history Keras для DL-моделей.
        """
        self.logger.info("Запуск оценки и визуализации результатов модели.")
        
        # Объединяем тестовый результат в один датафрейм для удобства
        evaluation_df = pd.DataFrame({
            'RUL_Actual': y_test,
            'RUL_Predicted': y_pred
        })

        evaluation_df = pd.merge(evaluation_df, X_test[['timestamp', 'bearing']], left_index=True, right_index=True, how='left')
        self.logger.info("График RUL будет построен по оси времени (timestamp) с разбивкой по подшипникам.")

        # Отладочный вывод для проверки данных перед графиком
        self.logger.debug(f"DEBUG: evaluation_df.head() before plotting:\n{evaluation_df.head()}")
        self.logger.debug(f"DEBUG: evaluation_df.tail() before plotting:\n{evaluation_df.tail()}")
        self.logger.debug(f"DEBUG: evaluation_df.dtypes before plotting:\n{evaluation_df.dtypes}")
        
        self._plot_scatter_comparison(evaluation_df)
        self._plot_rul_trend(evaluation_df)
        
        if self.model_type != 'DL':
            self._plot_feature_importance(feature_importance)
        if self.model_type == 'DL' and dl_history is not None:
            self._plot_training_history(dl_history)

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
        
        plt.title(f'Сравнение фактического RUL с предсказанным ({self.model_type})')
        plt.xlabel('Фактический RUL (часы)')
        plt.ylabel('Предсказанный RUL (часы)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = self.plots_dir / f'{self.model_type}_rul_scatter_comparison.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График сравнения RUL сохранен: {save_path}")
        
    def _plot_rul_trend(self, df: pd.DataFrame):
        """
        Строит график RUL в зависимости от времени, с разбивкой по подшипникам.
        """
        self.logger.info("Создание графика RUL во времени (по подшипникам)...")

        df = df.sort_values(by='timestamp')

        # Отладочный вывод для проверки данных, используемых для графика RUL-тренда
        for bearing in df['bearing'].unique():
            df_bearing = df[df['bearing'] == bearing]
            self.logger.debug(f"DEBUG: Data for RUL Trend Plot - Bearing: {bearing}")
            self.logger.debug(f"DEBUG: df_bearing.head() for RUL Trend:\n{df_bearing[['timestamp', 'RUL_Actual', 'RUL_Predicted']].head()}")
            self.logger.debug(f"DEBUG: df_bearing.tail() for RUL Trend:\n{df_bearing[['timestamp', 'RUL_Actual', 'RUL_Predicted']].tail()}")
            self.logger.debug(f"DEBUG: Min/Max Timestamp for {bearing}: {df_bearing['timestamp'].min()} / {df_bearing['timestamp'].max()}")
            self.logger.debug(f"DEBUG: Min/Max RUL_Actual for {bearing}: {df_bearing['RUL_Actual'].min()} / {df_bearing['RUL_Actual'].max()}")
            self.logger.debug(f"DEBUG: Min/Max RUL_Predicted for {bearing}: {df_bearing['RUL_Predicted'].min()} / {df_bearing['RUL_Predicted'].max()}")

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


        plt.title(f'Эволюция RUL: Фактический vs. Предсказанный ({self.model_type}) (Сглаживание: Окно={window})')
        plt.xlabel('Дата и время')
        plt.ylabel('RUL (часы)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = self.plots_dir / f'{self.model_type}_rul_trend_comparison.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График эволюции RUL сохранен: {save_path}")

    def _plot_training_history(self, history: Any):
        """
        Строит графики Loss и MAE по эпохам для DL-моделей.
        """
        self.logger.info("Создание графиков истории обучения (Loss/MAE)...")

        history_df = pd.DataFrame(history.history)
        
        # График Loss
        plt.figure(figsize=(12, 5))
        plt.plot(history_df['loss'], label='Train Loss')
        if 'val_loss' in history_df.columns:
            plt.plot(history_df['val_loss'], label='Validation Loss')
        
        plt.title(f'История обучения: Loss ({self.model_type})', fontsize=16)
        plt.xlabel('Эпоха', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path_loss = self.plots_dir / f'{self.model_type}_training_loss_history.png'
        plt.savefig(save_path_loss)
        plt.close()
        self.logger.info(f"График истории Loss сохранен: {save_path_loss}")

        # График MAE
        plt.figure(figsize=(12, 5))
        plt.plot(history_df['mae'], label='Train MAE')
        if 'val_mae' in history_df.columns:
            plt.plot(history_df['val_mae'], label='Validation MAE')
            
        plt.title(f'История обучения: MAE ({self.model_type})', fontsize=16)
        plt.xlabel('Эпоха', fontsize=12)
        plt.ylabel('MAE (часы)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path_mae = self.plots_dir / f'{self.model_type}_training_mae_history.png'
        plt.savefig(save_path_mae)
        plt.close()
        self.logger.info(f"График истории MAE сохранен: {save_path_mae}")

    def _plot_feature_importance(self, feature_importance: pd.Series, top_n: int = 20):
        """
        Строит график важности признаков (Feature Importance) для LightGBM.
        """
        self.logger.info(f"Создание графика TOP-{top_n} самых важных признаков.")
        
        # Берем топ-N
        top_features = feature_importance.head(top_n).sort_values(ascending=True)
        
        plt.figure(figsize=(10, 0.5 * top_n)) # Динамический размер графика
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
        
        plt.title(f'ТОП-{top_n} самых важных признаков для предсказания RUL ({self.model_type})', fontsize=14)
        plt.xlabel('Важность признака (Gain/Split)', fontsize=12)
        plt.ylabel('Признак', fontsize=12)
        plt.tight_layout()
        

        save_path = self.plots_dir / f'{self.model_type}_feature_importance_top20.png'
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"График важности признаков сохранен: {save_path}")