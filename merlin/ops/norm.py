import tensorflow as tf


def safe_norm(tensor,
              ord='euclidean',
              axis=None,
              keepdims=None,
              name=None):
    """
    Like tf.norm, but with the subgradient defined at 0.
    """
    output = tf.norm(
        tensor=tensor,
        ord=ord,
        axis=axis,
        keepdims=keepdims,
        name=name
    )

    if output.shape.rank == 0:
        # Rank zero tensors / scalars mess with the tf.where condition below.
        # Handle them specially here.
        return tf.cast(0., tensor.dtype) * output if float(output) == 0. else output

    valid_mask = output > 0
    return tf.scatter_nd(
        tf.where(valid_mask),
        output[valid_mask],
        output.shape
    )
