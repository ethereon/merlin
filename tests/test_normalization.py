import numpy as np
import tensorflow as tf
from numpy import testing

from merlin.modules.normalization import BatchNormalization, Normalization
from merlin.shape import Axis, TetraShape


class GaussianTestInput:
    mean = 42.
    stddev = 5.
    shape = TetraShape(n=3, h=10, w=20, c=6)
    tensor = tf.random.normal(
        shape=shape,
        mean=mean,
        stddev=stddev
    )


def all_axes_but_channel():
    return [axis for axis in range(4) if axis != Axis.channel]


def _test_batch_norm_output(batch_norm):
    batch_norm = BatchNormalization()
    output = batch_norm(GaussianTestInput.tensor)
    # Verify output is approximately zero mean
    testing.assert_almost_equal(
        tf.reduce_mean(output, axis=all_axes_but_channel()).numpy(),
        np.zeros((GaussianTestInput.shape.c)),
        decimal=5
    )
    # Verify output is approximately unit variance
    testing.assert_almost_equal(
        tf.math.reduce_std(output, axis=all_axes_but_channel()).numpy(),
        np.ones((GaussianTestInput.shape.c)),
        decimal=3
    )


def test_batch_normalized_output():
    _test_batch_norm_output(batch_norm=BatchNormalization())


def test_batch_normalized_params():
    batch_norm = BatchNormalization(fused=False)
    output = batch_norm(GaussianTestInput.tensor, training=True)
    scale = 1.0 - batch_norm.momentum
    # Verify moving mean (init = 0)
    testing.assert_allclose(
        batch_norm.moving_mean.numpy(),
        np.ones((GaussianTestInput.shape.c)) * GaussianTestInput.mean * scale,
        rtol=1e-2
    )
    # Verify moving variance (init = 1)
    testing.assert_allclose(
        np.sqrt((batch_norm.moving_variance.numpy() - batch_norm.momentum) / scale),
        np.ones((GaussianTestInput.shape.c)) * GaussianTestInput.stddev,
        rtol=1e-1
    )


def test_normalization_output():
    _test_batch_norm_output(batch_norm=Normalization('batch_norm'))
    _test_batch_norm_output(batch_norm=Normalization(BatchNormalization))
