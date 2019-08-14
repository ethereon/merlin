import functools
from contextlib import nullcontext
from typing import Callable, Iterable, Optional, Union

import tensorflow as tf

from merlin.modules.module import Module, module_name_from_type_name
from merlin.util.string import indent


def get_submodule_by_path(module, path):
    """
    Return a sub-module by its name.
    The path may refer to any nested node using '/' as the path delimiter.
    """
    # Separate out the next module's name
    parts = path.split('/', 1)
    name = parts[0]
    sub_path = parts[1] if len(parts) == 2 else None

    if hasattr(module, name):
        submodule = getattr(module, name)
        if sub_path is None:
            return submodule
        return get_submodule_by_path(module=submodule, path=sub_path)

    raise KeyError(f'No module named "{name}" found.')


class Sequential(Module):
    """
    Composes a sequence of unary submodules.
    Provides utilities like convenient member access and scoping.

    Scoping rules
    --------------
    A "scoped" Sequential instance allows enclosing submodules within a namescope.

    A Sequential instance is "scoped" if any of the following is true:
        - The scoped variable is explicitly set
        - An explicit name is provided

    For a scoped instance, the enclosing namescope is automatically entered:
        1. When adding instances passed during construction
        2. During invocation
    For case 1 above, be aware that the enclosing namescope only has effect if the
    submodule's construction is deferred. For instance, if the Sequential is constructed
    using a generator that creates and yields submodules, it will be properly scoped.
    However, passing a list of modules will have no effect on the (already constructed)
    submodules.

    The scope can also manually be entered at any time using a `with` statement.
    """

    def __init__(
        self,
        modules: Iterable[Module] = None,
        *,
        name: Optional[str] = None,
        scoped: Optional[bool] = None
    ):
        super().__init__(name=name)
        self._modules = []
        self._active_scope = None
        self._scoped = scoped if scoped is not None else (name is not None)

        if modules:
            with (self if self._scoped else nullcontext()):
                self.add_flattened(modules)

    def __call__(self, x):
        return super().__call__(x) if self._scoped else self.compute(x)

    def compute(self, x):
        """
        Sequentially compose contained modules.
        """
        for module in self._modules:
            x = module(x)
        return x

    def __enter__(self):
        """
        Activates the container's name scope.
        """
        assert self._active_scope is None
        if not self._scoped:
            raise RuntimeError('Attempted to scope using a non-scoped instance.')
        self._active_scope = tf.name_scope(self.name)
        self._active_scope.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        assert self._active_scope is not None
        self._active_scope.__exit__(*args, **kwargs)
        self._active_scope = None

    def add(self, *modules: Iterable[Module]):
        """
        Append the given modules to the chain.
        """
        for module in modules:
            if not hasattr(self, module.name):
                setattr(self, module.name, module)
            self._modules.append(module)
        return self

    def add_flattened(self, modules):
        """
        Flatten the potentially arbitrarily nested list of modules
        (eg: [a [b [c d] e] f] -> [a b c d e f]) and append them.

        Any "None" values are filtered out.
        """
        for module in modules:
            if callable(module):
                self.add(module)
            elif isinstance(module, Iterable):
                self.add_flattened(module)
            elif module is not None:
                raise ValueError(f'Invalid module type: {type( module)}')

    def __getitem__(self, key):
        """
        Return a contained module by either its index or name.
        """
        if isinstance(key, str):
            return get_submodule_by_path(module=self, path=key)

        if isinstance(key, int):
            return self._modules[key]

        raise ValueError(f'Invalid type for key: {type(key)}')

    def __iadd__(self, modules: Union[Callable, Iterable[Module]]):
        """
        Appends one or more modules.
        """
        if callable(modules):
            self.add(modules)
        else:
            self.add(*modules)
        return self

    def __truediv__(self, key):
        """
        Member access using pathlib style "/" separator.
        Eg: net / 'block_1' / 'conv2d'
        """
        # Disallow numeric keys for this operator to avoid accidental usage
        # in actual division contexts.
        assert isinstance(key, str)
        return self[key]

    def __iter__(self):
        yield from self._modules

    def __len__(self):
        return len(self._modules)

    def __str__(self):
        desc = super().__str__() + ' {\n'
        for module in self._modules:
            desc += indent(str(module), level=1) + '\n'
        desc += '}'
        return desc


class Residual(Module):

    def __init__(self, module, *, name=None):
        super().__init__(name=name)
        self.module = module

    def __call__(self, x):
        return x + self.module(x)


def chain(func) -> Callable[..., Sequential]:
    """
    A decorator for a function that returns a list of modules that
    should be chained/composed sequentially.

    This is equivalent to adding the modules in the returned list to
    a Sequential module, with the following tweaks:

        - The function call is nested within the sequential module's scope.

        - An optional keyword argument, "name", is injected. This is not
          forwarded to the wrapped function, but rather the Sequence module.
    """
    default_name = module_name_from_type_name(func.__name__)

    @functools.wraps(func)
    def chained(*args, name=default_name, **kwargs):
        with Sequential(name=name) as container:
            container.add_flattened(func(*args, **kwargs))
        return container
    return chained
