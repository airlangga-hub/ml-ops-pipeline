import logging
from logging.config import dictConfig
import os

LOG_DIR = os.path.join(os.getcwd(), "logs")

os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__file__)