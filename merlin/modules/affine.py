from typing import Optional

import tensorflow as tf

from merlin.initializers import Init
from merlin.modules.activation import Activation
from merlin.modules.keras import KerasAdapter
from merlin.spec import Spec


class Affine(KerasAdapter, tf.keras.layers.Dense):

    class Config(Spec):
        # Dimensionality of the output space
        units: int
        activation: Optional[Activation.Descriptor] = None
        # Whether to include the bias term
        use_bias: bool = True
        # The initializer to use for the weights
        kernel_initializer: Init.Descriptor = 'he_normal'
        # The initializer to use for the biases
        bias_initializer: Init.Descriptor = 'zeros'
