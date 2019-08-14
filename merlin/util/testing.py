import functools

import tensorflow as tf

from merlin.context import Context
from merlin.shape import TetraShape


def random_test_image(nhwc, dtype=tf.float32):
    return tf.random.uniform(
        shape=TetraShape(*nhwc).active_ordering,
        dtype=dtype
    )


def private_context(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with Context():
            return func(*args, **kwargs)
    return wrapped
