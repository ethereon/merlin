import shutil
from pathlib import Path
from typing import Callable

import tensorflow as tf

from merlin.spec import Spec
from merlin.util.logging import get_logger

logger = get_logger(__name__)


class Summarizer:

    class Config(Spec):
        # Write out a summary after this many iterations
        summary_period: int = 100
        # Path to a directory where the summary events will be written
        output_dir: str = None
        # If enabled, clears all prior content from the given output path
        clear_output_dir: bool = False

    def __init__(self, config: Config, callback: Callable):
        assert callable(callback)

        if config.clear_output_dir and Path(config.output_dir).exists():
            logger.info(f'Clearing summary output directory: {config.output_dir}')
            shutil.rmtree(config.output_dir)

        self.config = config
        self.callback = callback
        self.writer = tf.summary.create_file_writer(self.config.output_dir)
        self.call_count = 0

    def __call__(self):
        if self.call_count % self.config.summary_period == 0:
            with self.writer.as_default():
                self.callback()
        self.call_count += 1


def summarizer(config):
    def summarizer_decorator(func):
        return Summarizer(config=config, callback=func)
    return summarizer_decorator
