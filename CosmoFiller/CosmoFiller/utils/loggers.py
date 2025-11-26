import logging

def setup_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    """Crea un logger modulare con nome e formatter standard."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger