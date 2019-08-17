import tensorflow as tf

from merlin.shape import Axis


def global_average_pool(inputs, keep_dims=True):
    return tf.math.reduce_mean(
        inputs,
        axis=(Axis.height, Axis.width),
        keepdims=keep_dims
    )
