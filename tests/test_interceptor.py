from merlin.interceptor import OutputSelector
from merlin.modules.module import Module
from merlin.modules.util import Sequential
from merlin.util.testing import private_context


class ConstantAdder(Module):

    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value

    def compute(self, input):
        return self.value + input


@private_context
def get_simple_sequential():
    alpha = ConstantAdder(value=1., name='alpha')
    beta = ConstantAdder(value=2., name='beta')
    gamma = ConstantAdder(value=3., name='gamma')
    return alpha, beta, gamma, Sequential([alpha, beta, gamma])


def test_keyword_extraction():
    _, beta, _, composed = get_simple_sequential()

    with OutputSelector(beta=composed.beta) as selector:
        output = composed(42.)

    assert output == 48.
    assert selector[beta] == 45
    assert selector['beta'] == 45
    assert selector.pop('beta') == 45.


def test_positional_extraction():
    _, beta, _, composed = get_simple_sequential()

    with OutputSelector(composed.beta) as selector:
        output = composed(42.)

    assert output == 48.
    assert selector[beta] == 45
    assert selector[0] == 45
    assert selector.pop(0) == 45.


def test_ordered_extraction():
    alpha, beta, gamma, composed = get_simple_sequential()

    with OutputSelector(alpha, gamma, beta) as selector:
        composed(0.)

    expected_outputs = [
        1.,  # Alpha
        6.,  # Gamma
        3.,  # Beta
    ]

    assert list(selector.pop()) == expected_outputs
