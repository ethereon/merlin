def as_enumerable(maybe_enumerable):
    if isinstance(maybe_enumerable, (list, tuple)):
        yield from maybe_enumerable
    else:
        yield maybe_enumerable


def _flatten_yield(*args):
    for elem in args:
        # Explicitly restrict recursion to list and tuple [subclass]
        # instances, since Iterable/Sequence can unexpectedly lead
        # to unintended members being flattened.
        if isinstance(elem, (list, tuple)):
            yield from _flatten_yield(*elem)
        else:
            yield elem


def flatten(*args, generate=False):
    flattened = _flatten_yield(*args)
    return flattened if generate else list(flattened)
