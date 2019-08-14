class AbstractConfig:
    """
    Placeholder configuration type.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            'Must be replaced by subclass specific configuration class.'
        )

    def get(self, key):
        raise NotImplementedError()


def extract_config(cls, keyword_args):
    if (len(keyword_args) == 1):
        maybe_config = keyword_args.get('config')
        if isinstance(maybe_config, cls):
            # Constructed with an existing configuration
            return maybe_config

    # Constructed with configuration fields
    return cls(**keyword_args)


class Configurable:

    Config = AbstractConfig

    def __init__(self, **kwargs):
        self.configure(config=extract_config(cls=self.Config, keyword_args=kwargs))

    def configure(self, config):
        raise NotImplementedError()
