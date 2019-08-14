import tensorflow as tf
from numpy import testing

from merlin.context import active_context
from merlin.modules.convolution import Conv2D
from merlin.modules.convolution.padding import SamePadding
from merlin.shape import TetraShape
from merlin.util.testing import random_test_image


def _subsample(tensor, factor):
    if factor == 1:
        return tensor
    if active_context.is_channels_first:
        return tensor[:, :, ::factor, ::factor]
    return tensor[:, ::factor, ::factor, :]


def _test_same_align_pad_convolution(test_image, factor):
    # Convolve then subsample
    conv_then_subsample_output = Conv2D(
        filters=2,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=tf.initializers.constant(1.)
    )(test_image)
    conv_then_subsample_output = _subsample(conv_then_subsample_output, factor=factor)

    # Strided aligned convolution
    aligned_conv_output = Conv2D(
        filters=2,
        kernel_size=3,
        strides=factor,
        use_bias=False,
        padding=SamePadding(aligned=True),
        kernel_initializer=tf.initializers.constant(1.)
    )(test_image)
    testing.assert_array_equal(conv_then_subsample_output, aligned_conv_output)


def test_aligned_convolution_even():
    for stride in (1, 2):
        _test_same_align_pad_convolution(
            test_image=random_test_image(TetraShape(n=1, h=4, w=6, c=3)),
            factor=stride
        )


def test_aligned_convolution_odd():
    for stride in (1, 2):
        _test_same_align_pad_convolution(
            test_image=random_test_image(TetraShape(n=1, h=5, w=7, c=3)),
            factor=stride
        )
