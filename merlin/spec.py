import copy
import dataclasses
from collections.abc import Mapping
from types import MappingProxyType


class SpecError(Exception):
    pass


class Spec(Mapping):
    """
    A dataclass driven parameters container.

    There are a few distinctions between this spec container and regular dataclasses:

        - No additional decorators required.
          Inherited classes are automatically converted to dataclasses based Spec as well.

        - No unannotated fields allowed. This is strictly enforced.

        - Keyword args-only initialization for fields. This also works around the
          limitation in dataclasses which prevents classes with default values from
          being inherted.
    """

    def __init__(self, **kwargs):
        """
        Initializes a spec instance using the given keyword arguments
        corresponding to fields.
        """
        self._assign_fields(name_to_value=kwargs, allow_superset=False)

    def __init_subclass__(cls, **kwargs):
        """
        Auto-convert subclasses to dataclass.
        """
        decorated_class = dataclasses.dataclass(cls, init=False)
        assert decorated_class == cls
        cls._validate_fields()
        super().__init_subclass__(**kwargs)

    def _assign_fields(self, name_to_value: dict, allow_superset: bool):
        """
        Assign the fields of this spec using the given name_to_value dictionary.
        Unless :allow_superset: is true, asserts that all name_to_value pairs
        have been consumed.

        Note that the :name_to_value: dictionary is mutated by this method, with
        only the unassigned fields surviving after the call.
        """
        for field in self._fields:
            field_value = name_to_value.pop(field.name, field.default)
            if field_value is dataclasses.MISSING:
                if field.default_factory is not dataclasses.MISSING:
                    field_value = field.default_factory()
                else:
                    raise ValueError(
                        f'No value provided for field: {field.name}.\n'
                        f'Unparsed fields: {tuple(name_to_value.keys())}'
                    )
            setattr(self, field.name, field_value)

        if (not allow_superset) and name_to_value:
            raise ValueError(f'Unknown fields specified: {tuple(name_to_value.keys())}')

    def replace(self, **changes):
        """
        Returns a version of this spec with the given fields replaced.
        """
        return type(self).shallow_copy(spec=self, overrides=changes)

    def _transform_for_dict(self, obj):
        """
        Recursively transforms the given object for inclusion in the dict
        representation of the prop.
        """
        if isinstance(obj, Spec):
            return obj.as_dict()

        # Named tuples, detected in the same way as the official
        # dataclasses implementation.
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*(self._transform_for_dict(elem) for elem in obj))

        if isinstance(obj, (list, tuple)):
            return type(obj)(self._transform_for_dict(elem) for elem in obj)

        if isinstance(obj, dict):
            return type(obj)(
                (self._transform_for_dict(key), self._transform_for_dict(value))
                for key, value in obj.items()
            )

        return copy.deepcopy(obj)

    def as_dict(self, recursive=True):
        """
        Returns a (by default recursive) dictionary representation.
        """
        mapper = self._transform_for_dict if recursive else (lambda x: x)
        return {
            name: mapper(getattr(self, name))
            for name in self._field_names
        }

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError('Could not find a field named: ' + key) from err

    def __iter__(self):
        return self._field_names

    def __len__(self):
        return len(self._fields)

    def __contains__(self, item):
        return hasattr(self, item)

    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

    @property
    def _fields(self):
        return dataclasses.fields(self)

    @property
    def _field_names(self):
        return (field.name for field in self._fields)

    @classmethod
    def _validate_fields(cls):
        cls._check_for_missing_annotations()

    @classmethod
    def _is_candidate_for_field(cls, name):
        # Fields are expected to start with a lowercase letter
        return (name and name[0].islower() and not hasattr(Spec, name))

    @classmethod
    def _check_for_missing_annotations(cls):
        """
        By default, dataclass silently ignores any fields missing type
        annotations. For Spec subclasses, we guard against accidental
        omission of type annotations.
        """
        dataclass_field_names = set(field.name for field in dataclasses.fields(cls))
        all_plausible_field_names = set(filter(cls._is_candidate_for_field, dir(cls)))
        unannotated_fields = all_plausible_field_names - dataclass_field_names
        if unannotated_fields:
            raise SpecError(
                f'The class {cls.__name__} is missing type annotations '
                f'for the following fields: {unannotated_fields}'
            )

    @classmethod
    def from_superset(cls, **kwargs):
        """
        Create a spec instance from the given kwargs that's a superset of the
        required spec fields. Unlike the default constructor, this factory method
        ignores any additional fields.
        """
        spec = cls.__new__(cls)
        spec._assign_fields(name_to_value=kwargs, allow_superset=True)
        return spec

    @classmethod
    def shallow_copy(cls, spec, overrides: dict = None):
        # Use explicit assignment rather than invoking __init__ since we
        # want to support subclasses with custom constructors.
        name_to_value = dict(**spec)
        if overrides is not None:
            name_to_value.update(overrides)
        spec = cls.__new__(cls)
        spec._assign_fields(name_to_value=name_to_value, allow_superset=False)
        return spec

    @classmethod
    def frozen(cls, **kwargs):
        return MappingProxyType(cls(**kwargs))


class DynamicSpec(Spec):
    """
    Dynamic spec allow regular spec to be extended with an arbitrary
    set of fields during construction.
    """

    def __init__(self, **kwargs):
        # First assign all known fields (including defaults)
        self._assign_fields(name_to_value=kwargs, allow_superset=True)
        # Assign any remaining fields
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Save the dynamically provided field names
        self._dynamic_field_names = tuple(kwargs.keys())

    @property
    def _field_names(self):
        yield from super()._field_names
        yield from self._dynamic_field_names

    def __len__(self):
        return len(self._fields) + len(self._dynamic_field_names)


class Default(dataclasses.Field):
    """
    A default field helper.
    Useful when the default value is a mutable type such as a dict, list, etc.
    """

    def __init__(self, value=None, *, factory=None):
        if factory is None:
            assert value is not None
            if isinstance(value, type):
                # The given value is a type.
                # Use it as a nullary factory
                factory = value
            elif isinstance(value, Spec):
                factory = lambda: type(value).shallow_copy(value)
            else:
                constructor = type(value)
                factory = lambda: constructor(value)
        else:
            assert value is None
        super().__init__(
            default=dataclasses.MISSING,
            default_factory=factory,
            init=True,
            repr=True,
            hash=None,
            compare=None,
            metadata=None
        )
