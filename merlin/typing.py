from numbers import Number
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

Tensor = Union[tf.Tensor, np.ndarray]

# 2D spatial types
Height = Number
Width = Number
Size2D = Tuple[Height, Width]


def is_tensor(obj):
    return isinstance(obj, Tensor.__args__)
