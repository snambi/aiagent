import logging.config

logging_config = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"]
    }
}


def initialize_logger():
    logging.config.dictConfig(logging_config)
    
    logger = logging.getLogger(__name__)
    logger.info("Logger Initialized")