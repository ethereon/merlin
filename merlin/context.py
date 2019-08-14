import dataclasses
import functools
from enum import Enum, IntEnum

import tensorflow as tf

_g_context_stack = []


class ContextError(Exception):
    pass


class DataOrdering(str, Enum):
    NCHW = 'nchw'
    NHWC = 'nhwc'

    def to_keras(self):
        return 'channels_first' if self == 'NCHW' else 'channels_last'


class OperationMode(IntEnum):
    """
    The operation mode affects operations like batch normalization
    that behave differently in training and inference/test modes.

    The integer values below match Keras' "learning phase" numeric constants.
    """
    INFERENCE = 0
    TRAINING = 1


@dataclasses.dataclass
class Context:
    # Sets the default data orderig for operations such as
    # 2D convolutions, batch normalization, etc.
    # This also auto-sets the data ordering for the keras backend.
    data_ordering: DataOrdering = DataOrdering.NHWC

    # This affects operations like batch normalization that behave differently
    # in training and inference modes
    operation_mode: OperationMode = OperationMode.TRAINING

    # An arbitrary dictionary containing context-specific key-value pairs
    storage: dict = dataclasses.field(default_factory=dict)

    def push(self):
        """
        Make this the current context by pushing it on to the top of the stack.
        """
        _g_context_stack.append(self)
        self.on_activated()

    def pop(self, activate_prior=True):
        """
        Remove this context from the stack
        """
        if self not in _g_context_stack:
            raise ContextError('This context has not been pushed!')
        if not (_g_context_stack[-1] is self):
            raise ContextError('This context is not at the top of the stack!')
        _g_context_stack.pop()
        if activate_prior and _g_context_stack:
            _g_context_stack[-1].on_activated()

    def supersede(self):
        """
        Replace the previous context on the stack with this one.
        """
        if not _g_context_stack:
            raise ContextError('No prior context exists!')
        _g_context_stack[-1].pop(activate_prior=False)
        self.push()

    def on_activated(self):
        self.sync_keras_backend()

    def sync_keras_backend(self):
        tf.keras.backend.set_learning_phase(int(self.operation_mode))
        tf.keras.backend.set_image_data_format(self.data_ordering.to_keras())

    def replace(self, **changes):
        """
        Returns a version of this context with the given fields replaced.
        """
        return dataclasses.replace(self, **changes)

    @property
    def is_channels_first(self):
        return self.data_ordering == DataOrdering.NCHW

    @property
    def is_channels_last(self):
        return self.data_ordering == DataOrdering.NHWC

    @property
    def is_training_mode(self):
        is_training = self.operation_mode == OperationMode.TRAINING
        # Ensure Keras is in-sync.
        is_keras_training = (tf.keras.backend.learning_phase() == 1)
        if is_keras_training != is_training:
            raise ValueError(
                f'Keras learning phase is out of sync with the current context.\n'
                f'    Current context training : {is_training}\n'
                f'    Keras training phase     : {is_keras_training}'
            )
        return is_training

    @property
    def is_inference_mode(self):
        return not self.is_training_mode

    def in_inference_mode(self):
        return self.replace(operation_mode=OperationMode.INFERENCE)

    def __enter__(self):
        self.push()

    def __exit__(self, exc_type, exc, exc_tb):
        self.pop()


def get_active_context():
    if _g_context_stack:
        return _g_context_stack[-1]
    raise ContextError('No contexts available!')


class ActiveContextProxy:
    def __getattr__(self, name):
        return getattr(get_active_context(), name)


active_context = ActiveContextProxy()

# The default context
Context().push()


def inferential(func):
    @functools.wraps(func)
    def invoke_in_inference_mode(*args, **kwargs):
        if active_context.is_inference_mode:
            return func(*args, **kwargs)

        with active_context.in_inference_mode():
            return func(*args, **kwargs)

    return invoke_in_inference_mode
