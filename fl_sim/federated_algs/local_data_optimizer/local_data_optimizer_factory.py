from fl_sim import Status
from .random_optimizer import RandomOptimizer


class LocalDataOptimizerFactory:

    @staticmethod
    def get_optimizer(optimizer: str):
        if optimizer == "random":
            return RandomOptimizer()
