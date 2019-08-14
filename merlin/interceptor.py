from collections import defaultdict


class Interceptor:
    """
    Captures module calls and provides pre and post call hooks for subclasses.
    """

    def __init__(self, *modules):
        self.modules = modules

        # Group modules by their type
        self.module_type_to_instances = defaultdict(set)
        for module in modules:
            self.module_type_to_instances[type(module)].add(module)

    def _wrap_homogeneous_modules(self, kind, modules):
        """
        Wrap the given module's __call__ so that its invocations can be intercepted.
        """
        # Replace the __call__ class attribute (rather than the instance once), since
        # that's the one invoked by Python.
        original_call = kind.__call__

        def call_interceptor(module, *args, **kwargs):
            # Since this override is at the class level, only intercept calls
            # for the requested instances
            if module not in modules:
                return original_call(module, *args, **kwargs)

            self.pre_call(module, *args, **kwargs)
            output = original_call(module, *args, **kwargs)
            self.post_call(module, output, *args, **kwargs)
            return output

        # Stash the original __call__ for restoration
        call_interceptor._original_call = original_call
        # Stash the source interceptor for sanity checking
        call_interceptor._interceptor = self

        # Inject the call wrapper
        kind.__call__ = call_interceptor

    def _unwrap_homogeneous_modules(self, kind):
        """
        Restore the module's original __call__, ending its interception.
        """
        # Sanity check
        source_interceptor = getattr(kind.__call__, '_interceptor', None)
        if source_interceptor is None:
            raise RuntimeError('Internal inconsistency: non-wrapped module encountered.')
        if source_interceptor is not self:
            raise RuntimeError('Internal inconsistency: out-of-order module unwrapping detected.')

        # Restore the prior call
        kind.__call__ = kind.__call__._original_call

    def _wrap_modules(self):
        for kind, modules in self.module_type_to_instances.items():
            self._wrap_homogeneous_modules(kind=kind, modules=modules)

    def _unwrap_modules(self):
        for kind in self.module_type_to_instances:
            self._unwrap_homogeneous_modules(kind=kind)

    def pre_call(self, module, *args, **kwargs):
        """
        Invoked before calling the module.
        """

    def post_call(self, module, output, *args, **kwargs):
        """
        Invoked after calling the module.
        """

    def __enter__(self):
        self._wrap_modules()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self._unwrap_modules()


class OutputSelector(Interceptor):

    def __init__(self, *positional_modules, **keyword_modules):
        self.key_to_module = dict(enumerate(positional_modules))
        self.key_to_module.update(keyword_modules)
        if len(self.key_to_module) != (len(positional_modules) + len(keyword_modules)):
            raise ValueError('Duplicate modules detected.')

        # Dictionary that's populated on module invocation
        self.module_to_output = {}

        # Intercept the invocations of all given modules
        super().__init__(*self.key_to_module.values())

    def post_call(self, module, output, *args, **kwargs):
        self.module_to_output[module] = output

    def _get_output(self, key, remove=False):
        if key in self.key_to_module:
            # The key is a module name or index
            module = self.key_to_module[key]
        elif key in self.module_to_output:
            # The key is the module
            module = key
        else:
            raise KeyError(f'Invalid key: {key}')

        return self.module_to_output.pop(module) if remove else self.module_to_output[module]

    def __getitem__(self, key):
        return self._get_output(key, remove=False)

    def pop(self, key=None):
        if key is not None:
            return self._get_output(key, remove=True)

        # Gather all outputs and clear
        outputs = self.outputs
        self.key_to_output = {}
        return outputs

    @property
    def outputs(self):
        return [self.module_to_output.get(module) for module in self.key_to_module.values()]


class ComputeBreakpoint(Interceptor):
    """
    Triggers an ipdb breakpoint before the given module begins its computation.
    """

    def pre_call(self, *args, **kwargs):
        import ipdb
        ipdb.set_trace()
