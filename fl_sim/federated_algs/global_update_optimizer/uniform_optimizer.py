import numpy as np
from .global_update_optimizer import GlobalUpdateOptimizer
from ...status.orchestrator_status import OrchestratorStatus


class UniformOptimizer(GlobalUpdateOptimizer):

    def __init__(self, epochs: int, batch_size: int, num_examples: int, status: OrchestratorStatus, logger):
        super().__init__(epochs, batch_size, num_examples, status, logger)
        self.p_heterogeneity = 1
        self.epochs_min = 1
        self.epochs_max = epochs
        self.batch_size_min = batch_size
        self.batch_size_max = batch_size
        self.num_examples_min = 10
        self.num_examples_max = num_examples

    def optimize(self, r: int, dev_index: int, phase: str) -> dict:
        is_heterogeneous = np.random.binomial(1, self.p_heterogeneity)
        epochs = self.epochs_max
        batch_size = self.batch_size_max
        num_examples = self.num_examples_max
        if is_heterogeneous:

            if self.epochs_min < self.epochs_max:
                epochs = np.random.randint(low=self.epochs_min, high=self.epochs_max)

            if self.batch_size_min < self.batch_size_max:
                batch_size = np.random.randint(low=self.batch_size_min, high=self.batch_size_max)

            if self.num_examples_min < self.num_examples_max:
                num_examples = np.random.randint(low=self.num_examples_min, high=self.num_examples_max)

        return {"epochs": epochs, "batch_size": batch_size, "num_examples": num_examples}

