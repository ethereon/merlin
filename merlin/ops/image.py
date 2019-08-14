from functools import partial
from typing import Callable, Iterable

import tensorflow as tf

from merlin.context import active_context
from merlin.shape import (Axis,
                          get_spatial_size,
                          transpose_nchw_to_nhwc,
                          transpose_nhwc_to_nchw)
from merlin.typing import Tensor

# TensorFlow v1 resize operation with aligned corners (deprecated in v2).
# This is primarily useful for compatibility with legacy models.
legacy_aligned_resampler = partial(tf.compat.v1.image.resize, align_corners=True)


def resample(
    tensor,
    *,
    method: str,
    size: tuple = None,
    scale: float = None,
    like: Tensor = None,
    resampler: Callable = tf.image.resize
):
    """
    Resample the given rank 4 tensor using the given method.

    Exactly one of the following must be specified:
        size  : A (height, width) tuple for the output size
        scale : Either a scalar or a (scale_h, scale_w) tuple
        like  : Another rank 3 or 4 tensor. Theresampled output will have its spatial dimensions.

    The resampler may be substituted for an arbitrary callable whose API is compatible with
    that of the default TensorFlow resize operation.

    The data ordering (NCHW or NHCW) of all tensors is interpreted based on the current context.
    """
    # Get the input shape
    input_shape = tuple(tensor.shape)
    assert len(input_shape) == 4

    if size is not None:
        assert len(size) == 2
        output_size = size
    elif scale is not None:
        # Normalize scale to (scale_h, scale_w)
        if isinstance(scale, Iterable):
            assert len(scale) == 2
        else:
            scale = (scale, scale)
        # Compute the output shape
        output_size = (
            scale[0] * input_shape[Axis.height],
            scale[1] * input_shape[Axis.width]
        )
    elif like is not None:
        output_size = get_spatial_size(like)
    else:
        raise ValueError('An output shape related parameter must be specified.')

    # Pre-empt no-ops
    if output_size == get_spatial_size(tensor):
        return tensor

    # TensorFlow resize only operates on NHWC. Convert to NCHW.
    if active_context.is_channels_first:
        tensor = transpose_nchw_to_nhwc(tensor)

    # Resize
    resized = resampler(
        images=tensor,
        size=output_size,
        method=method
    )

    # Transpose back to the current ordering
    if active_context.is_channels_first:
        resized = transpose_nhwc_to_nchw(resized)

    return resized


resample.NEAREST_NEIGHBOR = tf.image.ResizeMethod.NEAREST_NEIGHBOR
resample.BILINEAR = tf.image.ResizeMethod.BILINEAR
resample.BICUBIC = tf.image.ResizeMethod.BICUBIC
resample.AREA = tf.image.ResizeMethod.AREA


def load_image(path: str) -> Tensor:
    return tf.io.decode_image(tf.io.read_file(path))


def write_image(path: str, image: Tensor):
    if path.endswith('.jpeg') or path.endswith('.jpg'):
        encoder = tf.image.encode_jpeg
    elif path.endswith('.png'):
        encoder = tf.image.encode_png
    else:
        raise ValueError(f'Unrecognized image extension in given path: {path}')

    # Remove any singleton dimensions
    if len(image.shape) == 4:
        image = tf.squeeze(image)

    tf.io.write_file(path, encoder(image))
