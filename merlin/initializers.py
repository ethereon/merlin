from typing import Union

import tensorflow as tf


class Init:
    """
    Weights initialization.

    TODO(saumitro): Flesh this out.
    """

    # Either the name of an initializer or an instance of a TensorFlow initializer.
    Descriptor = Union[str, tf.initializers.Initializer]
