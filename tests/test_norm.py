import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_equal

from merlin.gradient import compute_gradient
from merlin.ops.norm import safe_norm


def _random_partially_zero_tensor(rank):
    tensor = tf.random.uniform(shape=(4,) * rank).numpy()
    for axis in range(rank):
        # Set everything along index 1 of this axis to zero.
        indices = [slice(None)] * rank
        indices[axis] = 1
        tensor[tuple(indices)] = 0.
    return tf.constant(tensor)


def _test_safe_norm_forward(tensor):
    for axis in [None] + list(range(tensor.shape.rank)):
        assert_array_equal(
            safe_norm(tensor, axis=axis),
            tf.norm(tensor, axis=axis)
        )


def _test_safe_norm_gradients(tensor):
    for axis in [None] + list(range(tensor.shape.rank)):
        grad_safe_norm = compute_gradient(
            lambda: safe_norm(tensor, axis=axis),
            tensor
        ).numpy()
        grad_norm = compute_gradient(
            lambda: tf.norm(tensor, axis=axis),
            tensor
        ).numpy()
        assert grad_safe_norm.shape == grad_norm.shape

        # Make sure our gradient is finite everywhere
        assert np.all(tf.math.is_finite(grad_safe_norm))

        finite_mask = np.isfinite(grad_norm)

        # Make sure the gradient matches TensorFlow's finite values
        assert_array_equal(
            grad_safe_norm[finite_mask],
            grad_norm[finite_mask]
        )
        assert np.all(grad_safe_norm[np.invert(finite_mask)] == 0.)


def _test_safe_norm(tensor):
    _test_safe_norm_forward(tensor)
    _test_safe_norm_gradients(tensor)


def test_all_zeros():
    tensors = (
        tf.constant(0.),
        tf.zeros((1,)),
        tf.zeros((1, 2,)),
        tf.zeros((1, 2, 3, 4)),
        tf.zeros((1, 2, 3, 4, 5))
    )
    for tensor in tensors:
        _test_safe_norm(tensor)


def test_partially_zeros():
    for rank in range(5):
        _test_safe_norm(_random_partially_zero_tensor(rank=rank))
