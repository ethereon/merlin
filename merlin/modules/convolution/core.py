from typing import Optional, Union

import tensorflow as tf

from merlin.context import DataOrdering
from merlin.initializers import Init
from merlin.modules.activation import Activation
from merlin.modules.configurable import Composite
from merlin.modules.convolution.padding import ConvolutionPadding
from merlin.modules.keras import KerasAdapter
from merlin.spec import Spec
from merlin.typing import Size2D


class ConvBase(KerasAdapter):
    """
    Common behavior for Keras based convolution modules.
    """

    class Config(Spec):
        # The spatial size of the convolution kernel
        kernel_size: Union[int, Size2D]
        # The convolution stride
        strides: Union[int, Size2D] = 1
        # The padding to apply to the input
        padding: Union[str, ConvolutionPadding] = 'valid'
        # The data ordering
        data_format: Optional[DataOrdering] = None
        # The dilation rate
        dilation_rate: int = 1
        # An optional activation function to apply
        activation: Optional[Activation.Descriptor] = None
        # Whether the convolution includes a bias term
        use_bias: bool = True
        # Initializer for the bias vector
        bias_initializer: Optional[Init.Descriptor] = 'zeros'
        # An optional module name
        name: Optional[str] = None

    def __init__(self, **kwargs):
        # Check if we have custom padding
        padding = kwargs.get('padding')
        self._explicit_padding = padding if isinstance(padding, ConvolutionPadding) else None

        # Initialize via the keras adapter
        super().__init__(**kwargs)

        # Configure custom padding
        if self._explicit_padding:
            self._explicit_padding.setup(self)

    def adapt_for_keras(self, params):
        super().adapt_for_keras(params)

        # Disable base padding if it's explicitly handled
        if self._explicit_padding:
            params['padding'] = 'valid'

        # Create activations specified as configurations
        activation = params.get('activation')
        if isinstance(activation, Activation.Config):
            with tf.name_scope(params['name']):
                params['activation'] = Activation(**activation)

    def call(self, inputs):
        if self._explicit_padding is not None:
            inputs = self._explicit_padding(inputs)
        return super().call(inputs)


class Conv2D(ConvBase, tf.keras.layers.Conv2D):

    class Config(ConvBase.Config):
        # The number of output channels
        filters: int
        # Initializer for the kernel weights tensor
        kernel_initializer: Optional[Init.Descriptor] = 'he_normal'
        # Initializer for the bias vector
        bias_initializer: Optional[Init.Descriptor] = 'zeros'


class DepthwiseConv2D(ConvBase, tf.keras.layers.DepthwiseConv2D):

    class Config(ConvBase.Config):
        depthwise_initializer: Optional[Init.Descriptor] = 'he_normal'
        depth_multiplier: int = 1


class DepthwiseSeparableConv(Composite):
    """
    A depthwise convolution followed by a pointwise (1x1) convolution.
    """

    class Config(DepthwiseConv2D.Config):
        # The number of output channels produced by the pointwise convolution
        num_outputs: int
        # Initializer for the 1x1 pointwise convolution kernel
        pointwise_initializer: Optional[Init.Descriptor] = 'he_normal'

    def configure(self, config: Config):
        return (
            # Depthwise
            DepthwiseConv2D(
                **DepthwiseConv2D.Config.from_superset(**config).replace(
                    name='depthwise'
                ),
            ),

            # Pointwise
            Conv2D(
                kernel_size=1,
                strides=1,
                filters=config.num_outputs,
                use_bias=config.use_bias,
                activation=config.activation,
                kernel_initializer=config.pointwise_initializer,
                bias_initializer=config.bias_initializer,
                name='pointwise'
            )
        )
