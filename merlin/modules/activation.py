from typing import Callable, Optional, Union

import tensorflow as tf

from merlin.modules.module import Module
from merlin.modules.normalization import Normalization
from merlin.modules.util import Sequential
from merlin.spec import DynamicSpec, Spec


class ActivationProxy(Module):
    """
    Wraps an activation function in a module.
    """

    def __init__(self, activation: Callable, name=None, **kwargs):
        super().__init__(name=name)
        # An arbitrary callable activation function
        self.activation = activation
        # Zero or more keyword arguments to be forwarded to the activation function.
        # For instance, the fixed slope parameter for a leaky relu.
        self.activation_kwargs = kwargs

    def compute(self, features):
        return self.activation(features, **self.activation_kwargs)

    @classmethod
    def wrap(cls, activation):
        def activation_factory(**kwargs):
            return cls(activation=activation, **kwargs)
        return activation_factory


class Activation:

    class Config(DynamicSpec):
        """
        Partial configuration for the activation module.
        Any additional fields are forwarded to the specified activation module.
        """
        # The type of activation to use
        # Either a name of a known activation function (eg: 'relu', 'tanh', and all other
        # activations defined under tf.nn), or a factory function that returns an instance
        # of the activation module.
        kind: Union[str, Callable]

    def __new__(cls, kind: Union[str, Callable], **normalizer_kwargs):
        factory = kind if callable(kind) else cls.by_name(name=kind)
        return factory(**normalizer_kwargs)

    @classmethod
    def by_name(cls, name):
        if hasattr(tf.nn, name):
            return ActivationProxy.wrap(getattr(tf.nn, name))
        if hasattr(tf.keras.activations, name):
            return ActivationProxy.wrap(getattr(tf.keras.activations, name))
        raise ValueError(f'Unable to find activation named "{name}"')


class NormalizingActivation:
    """
    A factory class that creates configurations for composing an activation
    and a normalization function (eg: BatchNorm(ReLU(inputs))).

    Encapsulates nuances such as ordering and bias hints.
    """

    class Config(Spec):
        # Activation configuration
        activation: Optional[Activation.Config]
        # Normalization configuration
        normalization: Optional[Normalization.Config]
        # If true, the activation function is applied before normalization.
        activate_before_normalize: bool = False
        # Whether the given configuration is such that when composed with a
        # affine function it can absorb the bias term.
        # This parameter is non-functional as far as this activation is concerned,
        # and serves primarily as convenience flag for composition.
        absorbs_bias: Optional[bool] = None

    def __new__(cls, **kwargs):
        config = cls.Config(**kwargs)

        def normalizing_activation_factory():
            functions = (
                Normalization(**config.normalization)
                if config.normalization is not None
                else None,

                Activation(**config.activation)
                if config.activation is not None
                else None
            )
            if config.activate_before_normalize:
                functions = reversed(functions)

            return Sequential(functions)

        return Activation.Config(kind=normalizing_activation_factory)


# Type for specifying an activation function for commonly used modules like
# convolution and affine. One of the following:
# 1. Name of a known activation function
# 2. An arbitrary unary function
# 3. An activation configuration
Activation.Descriptor = Union[str, Callable, Activation.Config]
