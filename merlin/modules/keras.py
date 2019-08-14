"""
Utilities for inter-operability with Keras layers.
"""

from merlin.context import DataOrdering
from merlin.modules.module import Module, get_scope_unique_name_for_type


class KerasAdapter:
    """
    Mixin for modules derived from Keras layers.
    """

    def __init__(self, *args, **kwargs):
        # Generate a unique name
        kwargs['name'] = get_scope_unique_name_for_type(type(self), kwargs.get('name'))
        # Modify parameters to match Keras' API
        self.adapt_for_keras(params=kwargs)
        # Invoke the Keras baseclass
        super().__init__(*args, **kwargs)

    def adapt_for_keras(self, params):
        # Translate canonical data ordering to Keras' variant
        data_format = params.get('data_format')
        if isinstance(data_format, DataOrdering):
            params['data_format'] = data_format.to_keras()

    def __str__(self):
        return Module.__str__(self)
