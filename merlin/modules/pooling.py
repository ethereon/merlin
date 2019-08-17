from merlin.ops.pooling import global_average_pool
from merlin.modules.module import Module


class GlobalAveragePooling(Module):
    def compute(self, inputs):
        return global_average_pool(inputs)
