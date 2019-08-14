import tensorflow as tf

from merlin.modules.module import Module as BaseModule
from merlin.modules.util import Sequential
from merlin.util.configurable import AbstractConfig, extract_config


class Module(BaseModule):

    Config = AbstractConfig

    def __init__(self, **kwargs):
        config = extract_config(cls=self.Config, keyword_args=kwargs)
        # Invoke super to establish the module name
        super().__init__(name=config.get('name'))

        # Scoped configure
        with tf.name_scope(self.name):
            self.configure(config)

    def configure(self, config):
        raise NotImplementedError()


class Composite(Sequential):
    """
    A scoped sequential subclass that invokes inheritors with
    an initialized configuration instance.
    """

    Config = AbstractConfig

    def __init__(self, **kwargs):
        config = extract_config(cls=self.Config, keyword_args=kwargs)
        super().__init__(name=config.get('name'), scoped=True)
        with self:
            self.add_flattened(self.configure(config))

    def configure(self, config):
        raise NotImplementedError()
