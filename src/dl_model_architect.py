# src/dl_model_architect.py

import logging
from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Activation
from tensorflow.keras.layers import Attention, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class DLModelArchitect:
    """
    Класс для создания DL-модели (1D-CNN + LSTM + Attention) для RUL.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_cnn_lstm_attention(self, sequence_length: int, num_features: int, lstm_units: int, dropout_rate: float, learning_rate: float) -> Model:
        """
        Собирает DL-модель с 1D-CNN, LSTM и Attention.

        Args:
            sequence_length (int): T (Sequence Length).
            num_features (int): F (Number of Features).
            lstm_units (int): Количество юнитов в LSTM-слое.
            dropout_rate (float): Коэффициент отсева для слоев Dropout.
            learning_rate (float): Шаг обучения для компиляции модели.

        Returns:
            Model: Собранная модель Keras.
        """
        self.logger.info("Сборка DL-архитектуры: 1D-CNN-LSTM с Attention.")

        # 1. Входной слой: (T, F)
        inputs = Input(shape=(sequence_length, num_features))
        
        # 2. 1D CNN: Извлечение локальных паттернов (уменьшение Sequence Length)
        # 1D Conv позволяет моделировать локальные зависимости в фичах (например, между peak_1 и peak_2)
        x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='causal')(inputs)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)

        # 3. LSTM: Моделирование временной зависимости (T)
        # return_sequences=True требуется для следующего слоя Attention
        lstm_out = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(dropout_rate)(lstm_out)

        # 4. Attention Mechanism (Mechanism of Attention)
        # Query, Value, Key - все берутся из выхода LSTM
        attn_output = Attention()([lstm_out, lstm_out])

        # 5. GlobalAveragePooling1D: Агрегация временной оси после Attention
        # Сводит T*F выхода Attention к одному вектору для каждого примера
        x = GlobalAveragePooling1D()(attn_output) ## ДОБАВЛЕНО
        
        # 6. Промежуточный Dense слой: Обработка агрегированных признаков
        x = Dense(lstm_units, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        # 7. Выходной слой: Регрессия RUL
        # Линейная активация для регрессии
        outputs = Dense(1, activation=None, name='rul_output')(x)

        # 8. Компиляция
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # model.summary()
        self.logger.info("DL-Архитектура успешно собрана.")

        return model