from numbers import Number
from typing import Optional, Sequence, Callable
from functools import partial

import numpy as np
import tensorflow as tf

from merlin.typing import Tensor

# A matrix where each row is a single color
ColorMatrix = np.ndarray

# A palette is a callable that accepts a positive count and returns
# a color matrix with that many rows
Palette = Callable[[int], ColorMatrix]


def linearly_interpolate(colors: ColorMatrix, count: int):
    """
    Generates :count: colors from this palette, interpolating
    intermediate colors if necessary.

    TODO(saumitro): Lift interpolation to perceptual space
    """
    palette_size = len(colors)
    if count <= palette_size:
        return colors[:count]

    # Linearly interpolate the required number of color values
    max_palette_idx = palette_size - 1
    x = np.linspace(start=0, stop=max_palette_idx, num=count)
    idx_left = np.floor(x).astype(int)
    idx_right = np.minimum(np.floor(idx_left + 1), max_palette_idx).astype(int)
    alpha = (x - idx_left)[..., np.newaxis]
    interpolated = (1 - alpha) * colors[idx_left] + alpha * colors[idx_right]
    return interpolated.astype(colors.dtype)


def linearly_interpolate_separated(colors: ColorMatrix, count: int):
    """
    Generates :count: colors from this palette, interpolating
    intermediate colors if necessary.
    The interpolated colors sequentially appended at the end rather than
    placed in order.
    """
    palette_size = len(colors)
    if count <= palette_size:
        return colors[:count]

    num_missing = count - palette_size
    assert palette_size > 1
    num_subdivisions = int(np.ceil(num_missing / (palette_size - 1)))
    step = 1 / (num_subdivisions + 1)

    interpolated = [colors]
    for division in range(num_subdivisions):
        alpha = (division + 1) * step
        group = (1 - alpha) * colors[:-1] + alpha * colors[1:]
        interpolated.append(group)

    return np.concatenate(interpolated, axis=0)[:count].astype(colors.dtype)


# Color Brewer Qualitative 12-class Paired + Initial black
# from http://colorbrewer2.org
COLOR_BREWER_QUALITATIVE_PAIRED = np.array(
    (
        (166, 206, 227),
        (31, 120, 180),
        (178, 223, 138),
        (51, 160, 44),
        (251, 154, 153),
        (227, 26, 28),
        (253, 191, 111),
        (255, 127, 0),
        (202, 178, 214),
        (106, 61, 154),
        (255, 255, 153),
        (177, 89, 40),
    ),
    dtype=np.uint8
)

# Tableau 20 categorical
# from https://public.tableau.com
TABLEAU_20 = np.array(
    (
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
    ),
    dtype=np.uint8
)

DEFAULT_CATEGORICAL_PALETTE = partial(
    linearly_interpolate_separated,
    COLOR_BREWER_QUALITATIVE_PAIRED
)


class DiscreteColorMapper:

    def __init__(
        self,
        palette=DEFAULT_CATEGORICAL_PALETTE,
        input_space: Optional[Sequence[Number]] = None,
        input_cardinality: Optional[int] = None,
        zero_color: Optional[np.ndarray] = None
    ):
        # Establish the cardinality of the input space
        if input_cardinality is None:
            assert input_space is not None
            input_cardinality = len(input_space)
            # TODO(saumitro): Implement support for offset and non-contiguous input spaces.
            assert all(i == j for i, j in zip(input_space, range(input_cardinality)))
        elif input_space is not None:
            assert len(input_space) == input_cardinality

        # Generate the color lookup table
        self._lut = palette(input_cardinality - (1 if zero_color is not None else 0))

        # Inject the color for zero if provided
        if zero_color is not None:
            row = np.array(zero_color, dtype=self._lut.dtype)[np.newaxis]
            self._lut = np.concatenate([row, self._lut])

    def __call__(self, image: Tensor) -> Tensor:
        return tf.gather(self._lut, image)
