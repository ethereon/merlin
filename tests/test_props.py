from collections import namedtuple

import pytest

from merlin.spec import Spec, SpecError, DynamicSpec, Default


class Widget(Spec):
    name: str
    price: float
    version: int = 1
    features: dict = Default({'organic': True})


Pair = namedtuple('Pair', 'first second')


class WidgetSet(Spec):
    name: str
    widgets: tuple
    category: Pair
    manifest: dict


class DynamicWidget(DynamicSpec):
    name: str
    price: float
    version: int = 1
    features: dict = Default({'organic': True})


class DynamicWidgetSet(DynamicSpec):
    name: str
    widgets: tuple
    category: Pair
    manifest: dict


def test_defaults():
    widget = Widget(name='chronotron', price=3.14)
    assert widget.version == 1
    assert widget.features['organic']


def test_full_init():
    widget = Widget(
        name='spacebot',
        price=1337.,
        version=12,
        features={'organic': False, 'origin': 'Betelgeuse'}
    )
    assert widget.name == 'spacebot'
    assert widget.price == 1337.
    assert widget.version == 12
    assert not widget.features['organic']
    assert widget.features['origin'] == 'Betelgeuse'


def test_fields_validation():
    with pytest.raises(SpecError):
        class JankyWidget(Spec):
            name: str
            price: float
            version = 1  # This should trigger a validation error


def test_iteration():
    widget = Widget(name='spacebot', price=1337.)
    assert list(widget) == ['name', 'price', 'version', 'features']


def test_membership():
    widget = Widget(name='spacebot', price=1337.)
    assert 'name' in widget
    assert 'price' in widget
    assert 'version' in widget
    assert 'features' in widget
    assert 'random' not in widget


def test_get_item():
    widget = Widget(name='spacebot', price=1337.)
    assert widget['name'] == 'spacebot'
    assert widget['price'] == 1337.
    assert widget['version'] == 1
    assert widget['features'] == {'organic': True}


def test_mapping():
    widget = Widget(name='spacebot', price=1337.)
    mapping = dict(**widget)
    assert mapping == {
        'name': 'spacebot',
        'price': 1337.,
        'version': 1,
        'features': {'organic': True}
    }


def test_len():
    widget = Widget(name='spacebot', price=1337.)
    assert len(widget) == 4


def test_get():
    widget = Widget(name='spacebot', price=1337.)
    assert widget.get('name') == 'spacebot'
    assert widget.get('random', 42) == 42


def test_from_superset():
    widget = Widget.from_superset(
        name='spacebot',
        price=1337.,
        random='bear'
    )
    assert widget.name == 'spacebot'
    assert widget.price == 1337.
    assert 'random' not in widget


def test_recursive_dict():
    widget_set = WidgetSet(
        name='UberWidget',
        widgets=(
            Widget(name='alpha', price=1.),
            Widget(name='beta', price=2., version=3)
        ),
        category=Pair(
            first=Widget(name='gamma', price=3.),
            second=42
        ),
        manifest={1: 2}
    )
    assert widget_set.as_dict() == dict(
        name='UberWidget',
        widgets=(
            {
                'name': 'alpha',
                'price': 1.,
                'version': 1,
                'features': {'organic': True}
            },
            {
                'name': 'beta',
                'price': 2.,
                'version': 3,
                'features': {'organic': True}
            },
        ),
        category=Pair(
            first={
                'name': 'gamma',
                'price': 3.,
                'version': 1,
                'features': {'organic': True}
            },
            second=42
        ),
        manifest={1: 2}
    )


def test_replacement():
    widget_src = Widget(name='source', price=1., version=3)
    widget_dst = widget_src.replace(name='dest', price=2.)
    # Verify source remains the same
    assert widget_src.name == 'source'
    assert widget_src.price == 1.
    # Verify replacements
    assert widget_dst.name == 'dest'
    assert widget_dst.price == 2.
    assert widget_dst.version == 3


def test_dynamic_spec_creation_with_no_extra_fields():
    widget = DynamicWidget(name='spacebot', price=42.)
    assert widget.name == 'spacebot'
    assert widget.price == 42.
    assert widget.version == 1
    assert widget.features == {'organic': True}
    assert len(widget) == 4
    assert list(widget) == ['name', 'price', 'version', 'features']


def test_dynamic_spec_creation_with_extra_fields():
    widget = DynamicWidget(
        name='spacebot',
        price=42.,
        dynamic_1='bear',
        dynamic_2=-1
    )
    assert widget.name == 'spacebot'
    assert widget.price == 42.
    assert widget.version == 1
    assert widget.features == {'organic': True}
    assert widget.dynamic_1 == 'bear'
    assert widget.dynamic_2 == -1
    assert len(widget) == 6
    assert 'name' in widget
    assert 'version' in widget
    assert 'dynamic_1' in widget
    assert 'dynamic_2' in widget
    assert list(widget) == [
        'name', 'price', 'version', 'features', 'dynamic_1', 'dynamic_2'
    ]


def test_dynamic_recursive_dict():
    widget_set = DynamicWidgetSet(
        name='UberWidget',
        widgets=(
            Widget(name='alpha', price=1.),
            DynamicWidget(
                name='beta',
                price=2.,
                version=3,
                arbitrary_1=1,
                arbitrary_2=Widget(name='sub_beta', price=2.)
            )
        ),
        category=Pair(
            first=DynamicWidget(name='gamma', price=3., axis=None),
            second=42
        ),
        manifest={1: 2}
    )
    assert widget_set.as_dict() == dict(
        name='UberWidget',
        widgets=(
            {
                'name': 'alpha',
                'price': 1.,
                'version': 1,
                'features': {'organic': True}
            },
            {
                'name': 'beta',
                'price': 2.,
                'version': 3,
                'features': {'organic': True},
                'arbitrary_1': 1,
                'arbitrary_2': {
                    'name': 'sub_beta',
                    'price': 2.,
                    'version': 1,
                    'features': {'organic': True}
                }
            },
        ),
        category=Pair(
            first={
                'name': 'gamma',
                'price': 3.,
                'version': 1,
                'features': {'organic': True},
                'axis': None
            },
            second=42
        ),
        manifest={1: 2}
    )
