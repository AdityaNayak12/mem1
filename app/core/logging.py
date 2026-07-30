import logging
import sys

# Configure a simple logger for the application
logger = logging.getLogger("mem1")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if logger is already configured
if not logger.handlers:
    handler = sys.stderr
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
