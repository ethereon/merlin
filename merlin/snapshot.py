from typing import Optional

import tensorflow as tf

from merlin.spec import Spec
from merlin.util.logging import get_logger

logger = get_logger(__name__)


class Snapshot:

    class Config(Spec):
        # Path to a directory where the snapshots will be saved
        directory: str
        # How frequently the snapshots will be saved
        interval: int = 1000
        # If set, only this many most recent snapshots will be saved.
        max_to_keep: Optional[int] = None
        # If set, snapshots spaced apart by this many hours will be
        # preserved (even if max_to_keep is set).
        preserve_every_n_hours: Optional[int] = None

    def __init__(self, config: Config, **trackables):
        self.config = config
        self.checkpoint = tf.train.Checkpoint(**trackables)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.config.directory,
            max_to_keep=self.config.max_to_keep,
            keep_checkpoint_every_n_hours=self.config.preserve_every_n_hours
        )
        self.num_iterations = 0

    def __call__(self):
        self.num_iterations += 1
        if (self.num_iterations % self.config.interval == 0):
            self.save()

    @property
    def latest(self):
        return self.checkpoint_manager.latest_checkpoint

    def save(self):
        ckpt_path = self.checkpoint_manager.save()
        logger.info(f'Saved snapshot: {ckpt_path}')

    def restore(self, snapshot=None):
        if snapshot is None:
            snapshot = self.latest

        if snapshot is None:
            raise ValueError('No snapshot provided or available for restoration.')

        self.checkpoint.restore(snapshot)
        logger.info(f'Restored checkpoint from {snapshot}')

    def maybe_restore_latest(self):
        if self.latest is not None:
            self.restore(snapshot=self.latest)


def load_snapshot(directory: str, **trackables):
    Snapshot(Snapshot.Config(directory=directory), **trackables).restore()
