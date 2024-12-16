from .common_imports import *
from .io import *
from .cv_utils import *
from .transforms import *
from .mano_info import NEW_MANO_FACES, NUM_MANO_VERTS


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def get_logger(log_name="HOCapToolkit", log_level="INFO", log_file=None):
    """Create and return a logger with console and optional file output."""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s:%(funcName)s] [%(levelname).3s] %(message)s",
        datefmt="%Y%m%d;%H:%M:%S",
    )
    if not logger.hasHandlers():
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
