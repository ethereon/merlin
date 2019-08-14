import logging

from merlin.util import console

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR


class Formatter(logging.Formatter):
    """
    ANSI colored log message formatter.
    """
    # Maps logging levels to their display strings
    LOG_LEVEL_DISPLAY = {
        DEBUG: ('D', console.cyan),
        INFO: ('I', console.green),
        WARNING: ('W', console.yellow),
        ERROR: ('E', console.red)
    }

    def format(self, record):
        # Formatted message
        msg = super().format(record)

        # Message level
        level_indicator, level_color = self.LOG_LEVEL_DISPLAY[record.levelno]
        level_msg = level_color(level_indicator)

        # Message time
        time = self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')
        time_color = level_color if record.levelno > INFO else console.dim
        time_msg = time_color(time)

        return ' '.join((level_msg, time_msg, msg))


def _setup_core_logger():
    logger = logging.getLogger('')
    logger.setLevel(INFO)

    # Remove any previously installed handlers
    # (TensorFlow may inject abseil loggers.)
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)

    # Setup stream handler
    handler = logging.StreamHandler()
    handler.setFormatter(Formatter())
    logger.addHandler(handler)

    return logger


_logger = _setup_core_logger()


def get_logger(name):
    return _logger.getChild(name)
