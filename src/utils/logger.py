import logging


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
