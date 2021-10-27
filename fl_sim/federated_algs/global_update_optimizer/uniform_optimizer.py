import numpy as np
from .global_update_optimizer import GlobalUpdateOptimizer
from ...status.orchestrator_status import OrchestratorStatus


class UniformOptimizer(GlobalUpdateOptimizer):

    def __init__(self, epochs: int, batch_size: int, num_examples: int, status: OrchestratorStatus, logger):
        super().__init__(epochs, batch_size, num_examples, status, logger)
        self.epochs_mean = epochs
        self.epochs_var = 2
        self.batch_size = batch_size
        self.batch_size_var = 1
        self.num_examples = num_examples
        self.num_examples_var = 1

    def optimize(self, r: int, dev_index: int, phase: str) -> dict:
        return {"epochs": np.random.randint(low=self.epochs_mean-self.epochs_var, high=self.epochs_mean+self.epochs_var),
                "batch_size": np.random.randint(low=self.batch_size-self.batch_size_var, high=self.batch_size+self.batch_size_var),
                "num_examples": np.random.randint(low=self.num_examples-self.num_examples_var, high=self.num_examples+self.num_examples_var)}


