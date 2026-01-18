import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    app_logger = logging.getLogger("src")
    app_logger.setLevel(level)

    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(level)


def add_file_handler(log_file: Path, level: int = logging.DEBUG) -> logging.FileHandler:
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    app_logger = logging.getLogger("src")
    app_logger.addHandler(file_handler)
    return file_handler


def remove_file_handler(handler: logging.FileHandler) -> None:
    app_logger = logging.getLogger("src")
    app_logger.removeHandler(handler)
    handler.close()
