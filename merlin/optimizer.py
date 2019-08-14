import tensorflow as tf

from merlin.spec import Spec, Default


class Optimizer:

    class Config(Spec):
        # An optimizer name from the tf.optimizers module
        name: str = 'RMSprop'
        # Keyword arguments passed to the optimizer on construction
        params: dict = Default(dict(lr=0.001))

    def __new__(cls, config: Config):
        return cls.get_optimizer_by_name(config.name)(**config.params)

    @classmethod
    def get_optimizer_by_name(cls, name):
        return getattr(tf.optimizers, name)
