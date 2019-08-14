import tensorflow as tf

from merlin.util.collections import as_enumerable


def compute_gradient(func, sources):
    with tf.GradientTape() as tape:
        for source in as_enumerable(sources):
            tape.watch(source)
        targets = func()
    return tape.gradient(targets, sources)
