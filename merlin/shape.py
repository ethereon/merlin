from collections import namedtuple
from typing import Sequence, Union

import numpy as np
import tensorflow as tf

from merlin.context import active_context
from merlin.typing import Tensor, is_tensor


def nhwc_to_nchw(shape):
    assert len(shape) == 4
    n, h, w, c = shape
    return (n, c, h, w)


def nchw_to_nhwc(shape):
    assert len(shape) == 4
    n, c, h, w = shape
    return (n, h, w, c)


def nhwc_to_active(shape):
    """
    Given an NHWC shape, convert it to the current ordering.
    """
    return shape if active_context.is_channels_last else nhwc_to_nchw(shape)


def nchw_to_active(shape):
    """
    Given an NHWC shape, convert it to the current ordering.
    """
    return shape if active_context.is_channels_first else nchw_to_nhwc(shape)


def transpose_nchw_to_nhwc(tensor):
    return tf.transpose(tensor, (0, 2, 3, 1))


def transpose_nhwc_to_nchw(tensor):
    return tf.transpose(tensor, (0, 3, 1, 2))


class TetraShape(namedtuple('TetraShapeBase', 'n h w c')):
    """
    A named tuple representing the shape of a rank 4 tensor in NHWC ordering.

    The preferred way of creating one is using keyword arguments:
        TetraShape(n=1, h=2, w=3, c=4) == TetraShape(n=1, c=4, h=2, w=3) == (1, 2, 3, 4)
    """

    @property
    def active_ordering(self):
        return nhwc_to_active(self)

    @property
    def nchw(self):
        return nhwc_to_nchw(self)


class ImageSize(namedtuple('ImageSizeBase', 'height width')):
    """
    A named tuple representing the shape of an image in hw ordering.
    """

    def scale(self, factor: float, quantize=None):
        scaled = np.array(self, dtype=np.float32) * factor
        if quantize is not None:
            scaled = np.array(quantize(scaled)).astype(int)
        return ImageSize(*scaled)


class AxisMeta(type):
    @property
    def batch(cls):
        return 0

    @property
    def channel(cls):
        return 1 if active_context.is_channels_first else 3

    @property
    def height(cls):
        return 2 if active_context.is_channels_first else 1

    @property
    def width(cls):
        return 3 if active_context.is_channels_first else 2


class Axis(metaclass=AxisMeta):
    pass


def get_spatial_size(obj: Union[Tensor, Sequence[int]]) -> ImageSize:
    """
    Given either a rank 4 or 3 tensor or its shape, returns the spatial size
    (height, width)  by interpreting the shape under the currently active data ordering.
    """
    shape = obj.shape if is_tensor(obj) else obj
    rank = len(shape)
    if rank == 4:
        return ImageSize(height=shape[Axis.height], width=shape[Axis.width])
    if rank == 3:
        assert Axis.batch == 0
        return ImageSize(height=shape[Axis.height - 1], width=shape[Axis.width - 1])

    raise ValueError(f'Invalid tensor rank: {rank}')
