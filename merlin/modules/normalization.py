from typing import Callable, Optional, Union

import tensorflow as tf

from merlin.initializers import Init
from merlin.modules.keras import KerasAdapter
from merlin.shape import Axis
from merlin.spec import DynamicSpec, Spec


class BatchNormalization(KerasAdapter, tf.keras.layers.BatchNormalization):

    class Config(Spec):
        # The axis along which the normalization will be performed.
        # If unspecified, the active context's channel axis is used.
        axis: Optional[int] = None

        # Momentum for the moving average.
        momentum: float = 0.99
        # Small float added to variance to avoid dividing by zero.
        epsilon: float = 1e-3
        # Whether to include the bias term "beta"
        center: bool = True
        # Wheter to include the scaling term "gamma"
        scale: bool = True
        # Bias initializer
        beta_initializer: Init.Descriptor = 'zeros'
        # Scale initializer
        gamma_initializer: Init.Descriptor = 'ones'
        # Moving mean initializer
        moving_mean_initializer: Init.Descriptor = 'zeros'
        # Moving variance initializer
        moving_variance_initializer: Init.Descriptor = 'ones'

        # Whether to use Batch Renormalization
        # See: https://arxiv.org/abs/1702.03275
        # This adds extra variables during training.
        # Inference remains the same.
        renorm: bool = False

        # A dictionary that may map keys {rmax, rmin, dmax} to
        # scalar Tensors used to clip the renorm correction. The correction
        # (r, d) is used as:
        #       corrected_value = normalized_value * r + d
        # with r clipped to [rmin, rmax], and d to [-dmax, dmax].
        # Missing {rmax, rmin, dmax} are set to {inf, 0, inf} respectively.
        renorm_clipping: Optional[dict] = None

        # Momentum used to update the moving means and standard
        # deviations with renorm. Unlike `momentum`, this affects training
        # and should be neither too small (which would add noise) nor too large
        # (which would give stale estimates). Note that `momentum` is still applied
        # to get the means and variances for inference.
        renorm_momentum: float = 0.99

        # Whether to use the (faster) fused batch normalization implementation.
        # If None, uses the fused implementation whenever possible.
        fused: Optional[bool] = None

        # Whether the batch norm parameters are "trainable".
        # This also switches the batch norm to inference mode.
        trainable: bool = True

        # By default, `virtual_batch_size` is `None`,
        # which means batch normalization is performed across the whole batch. When
        # `virtual_batch_size` is not `None`, instead perform "Ghost Batch
        # Normalization", which creates virtual sub-batches which are each
        # normalized separately (with shared gamma, beta, and moving statistics).
        # Must divide the actual batch size during execution.
        virtual_batch_size: Optional[int] = None

        # A function taking the Tensor containing the (dynamic) shape of
        # the input tensor and returning a pair (scale, bias) to apply to the
        # normalized values (before gamma and beta), only during training.
        # For example, if axis is -1, then:
        #   adjustment = lambda shape: (
        #       tf.random.uniform(shape[-1:],  0.93, 1.07),
        #       tf.random.uniform(shape[-1:],  -0.1,  0.1))
        # will scale the normalized value by up to 7% up or down, then shift the
        # result by up to 0.1 (with independent scaling and bias for each feature
        # but shared across all examples), and finally apply gamma and/or beta.
        # If None, no adjustment is applied.
        # Cannot be specified if virtual_batch_size is specified.
        adjustment: Optional[Callable] = None

        # An optional module name
        name: Optional[str] = None

    def __init__(self, *args, **kwargs):
        config = self.Config(*args, **kwargs)
        if config.axis is None:
            # Auto-set the normalization axis based on the currently active context
            config.axis = Axis.channel
        super().__init__(**config)


class Normalization:

    # Mapping of supported normalizer layer names to types
    _NAME_TO_NORMALIZER = {
        'batch_norm': BatchNormalization,
        'batch_normalization': BatchNormalization
    }

    class Config(DynamicSpec):
        """
        Partial configuration for a normalization layer.
        Any additional fields are forwarded to the specified normalization layer.
        """
        # The type of normalization to use
        kind: Union[str, Callable]

    def __new__(cls, kind: Union[str, Callable], **normalizer_kwargs):
        factory = kind if callable(kind) else cls.by_name(name=kind)
        return factory(**normalizer_kwargs)

    @classmethod
    def by_name(cls, name):
        """
        Returns the normalization module corresponding to the given name.
        Raises a ValueError if no matching module is found.
        """
        try:
            return cls._NAME_TO_NORMALIZER[name]
        except KeyError as err:
            raise ValueError(f'Unknown normalizer: {name}') from err
