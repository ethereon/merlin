import pytest
import tensorflow as tf

from merlin.modules.module import Module
from merlin.modules.util import Sequential
from merlin.util.testing import private_context


class ConstantAdder(Module):

    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value
        self.variable = None

    def compute(self, input):
        if self.variable is None:
            # A contrived variable to test scoping
            self.variable = tf.Variable(self.value, name='value')
        return self.variable + input


def test_basic():
    with Sequential(name='composite') as seq:
        seq += ConstantAdder(value=42.)
        seq += ConstantAdder(value=16.)
        seq += [ConstantAdder(value=1.), ConstantAdder(value=2.)]
    assert seq(12.).numpy() == 73.


@private_context
def test_scoping():
    with Sequential(name='alpha') as seq:
        adder = ConstantAdder(value=42., name='beta')
        seq += adder
    seq(0.)
    assert adder.variable.name == 'alpha/beta/value:0'


@private_context
def test_scoping_explicit():
    with Sequential(scoped=True) as seq:
        adder = ConstantAdder(value=42., name='beta')
        seq += adder
    seq(0.)
    assert adder.variable.name == 'sequential/beta/value:0'


@private_context
def test_no_scoping():
    adder = ConstantAdder(value=42., name='beta')
    seq = Sequential([adder])
    seq(0.)
    assert adder.variable.name == 'beta/value:0'


@private_context
def test_no_scoping_guard():
    with pytest.raises(RuntimeError):
        with Sequential() as seq:
            seq += ConstantAdder(value=42., name='beta')


@private_context
def test_accessors():
    alpha = ConstantAdder(value=42., name='alpha')
    beta = ConstantAdder(value=16., name='beta')
    seq = Sequential([alpha, beta])

    assert len(seq) == 2
    assert seq[0] is alpha
    assert seq[1] is beta
    assert seq['alpha'] is alpha
    assert seq['beta'] is beta
    assert seq / 'alpha' is alpha
    assert seq / 'beta' is beta
    assert seq.alpha is alpha
    assert seq.beta is beta


def test_iteration():
    alpha = ConstantAdder(value=42.)
    beta = ConstantAdder(value=16.)
    seq = Sequential([alpha, beta])
    assert list(seq) == [alpha, beta]


def test_add_flattened():
    a = ConstantAdder(value=1.)
    b = ConstantAdder(value=2.)
    c = ConstantAdder(value=3.)
    d = ConstantAdder(value=4.)
    e = ConstantAdder(value=5.)
    f = ConstantAdder(value=6.)
    seq = Sequential()
    seq.add_flattened([a, [b, [c, d], e], f])
    assert list(seq) == [a, b, c, d, e, f]


@private_context
def test_nested_access():
    with Sequential(name='alpha') as alpha:
        with Sequential(name='beta') as beta:
            with Sequential(name='gamma') as gamma:
                leaf = ConstantAdder(value=42., name='leaf')
                gamma += leaf
            beta += gamma
        alpha += beta

    alpha(0.)

    assert alpha['beta/gamma/leaf'] == leaf
    assert leaf.variable.name == 'alpha/beta/gamma/leaf/value:0'
