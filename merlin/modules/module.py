from typing import Dict, Optional

import tensorflow as tf

from merlin.context import active_context
from merlin.util.string import camel_to_snake


def get_scoped_name_to_index_map() -> Dict[str, int]:
    """
    A context-specific dictionary that maps scoped module names to their
    current max index. This is used for auto-suffixing modules with the same name.
    """
    return active_context.storage.setdefault('merlin/scope-map', {})


def reserve_scoped_index(name: str) -> int:
    """
    Given a non-scoped module name, reserves and returns a new index (starting at 0)
    for it within the current TensorFlow name scope and Merlin context.
    """
    cur_scope_name = tf.python.context.context().scope_name
    scoped_key = (cur_scope_name, name)
    scoped_name_to_index = get_scoped_name_to_index_map()
    new_index = scoped_name_to_index.get(scoped_key, -1) + 1
    scoped_name_to_index[scoped_key] = new_index
    return new_index


def get_scope_unique_name(name: str) -> str:
    """
    Given a potentially non-unique, non-scoped name, reserves and returns unique (within the
    current Merlin context) version of that name. If no other module with this name
    exists in this context, its returned unchanged. If this name has been previously
    reserved, its suffixed with a monotonically increasing index.
    """
    new_index = reserve_scoped_index(name)
    return name if new_index == 0 else (name + '_' + str(new_index))


def module_name_from_type_name(type_name: str) -> str:
    # Convert to snake case + remove _ prefixes
    return camel_to_snake(type_name).lstrip('_')


def get_scope_unique_name_for_type(module_type: type, name: Optional[str] = None) -> str:
    """
    Returns a scope-unique version of the given name. If the given name is None,
    a name derived from the given module's type (eg: SoftMax -> soft_max_2) is used.
    """
    return get_scope_unique_name(
        name or
        getattr(
            module_type,
            'INSTANCE_NAME_PREFIX',
            module_name_from_type_name(module_type.__name__)
        )
    )


class Module(tf.Module):
    """
    Base class for modules that provides the following additional features
    on top of the tf.Module implementation:

        - Name management (auto-generated from type name, auto-suffixed with indices)
        - Indirection for invocation that wraps the call in the module's name scope
        - Readable string representation
        - Propagation of trainable state

    See the design document for a discussion on various nuances involving this base class
    (as well as TensorFlow modules in general).
    """

    def __init__(self, *, name=None):
        super().__init__(name=self.generate_module_name(name))

    def compute(self, *args, **kwargs):
        raise NotImplementedError()

    def set_trainable(self, is_trainable: bool):
        for module in self.submodules:
            if hasattr(module, 'trainable'):
                module.trainable = is_trainable

    @classmethod
    def generate_module_name(cls, name: Optional[str] = None):
        return get_scope_unique_name_for_type(cls, name=name)

    def __call__(self, *args, **kwargs):
        """
        The default implementation nests the :compute: call within
        the Module's namescope. This results in deferred variables and submodules
        having this module's namescope as a prefix.

        Subclasses can safely override this method when this behavior is not desirable.
        """
        with tf.name_scope(self.name):
            return self.compute(*args, **kwargs)

    def __str__(self):
        """
        By default, displays the type and name of the module.
        """
        description = self.__class__.__name__
        name = getattr(self, 'name', None)
        if name is not None:
            description = name + ': ' + description
        return description
