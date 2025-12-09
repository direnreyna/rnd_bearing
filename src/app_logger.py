# src/app_logger.py

import logging
import pathlib
import sys

class AppLogger:
    """Класс для настройки и предоставления инстанса логгера."""

    @staticmethod
    def get_logger(name: str, log_filepath: pathlib.Path) -> logging.Logger:
        """
        Настраивает и возвращает логгер.

        Логгер будет выводить сообщения уровня INFO и выше
        одновременно в консоль (stdout) и в указанный файл.

        Args:
            name (str): Имя логгера (обычно __name__).
            log_filepath (pathlib.Path): Путь к файлу для записи логов.

        Returns:
            logging.Logger: Сконфигурированный экземпляр логгера.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Предотвращаем дублирование хендлеров, если логгер уже был создан
        if logger.hasHandlers():
            return logger

        # Форматтер для сообщений
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Хендлер для вывода в консоль
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Хендлер для записи в файл
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger