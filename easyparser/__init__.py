import logging
import os

from .auto import parse

logger = logging.getLogger(__name__)
logger.propagate = False

if debug_level := os.getenv("CHUNKING_DEBUG"):
    # DEBUG level "info" is meant for normal users to see the program progress.
    # DEBUG level "debug" is meant for developers to observe more verbose runtime
    # information to debug problems.
    debug_level = debug_level.lower().strip()
    if debug_level == "info":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

# set formatting
formatter = logging.Formatter(
    "[%(levelname)s][easyparser] %(asctime)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f"[:-3],  # This truncates microseconds to 3 digits
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


__all__ = ["parse"]
