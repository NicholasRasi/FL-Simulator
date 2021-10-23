import random
import numpy as np
from .global_update_optimizer import GlobalUpdateOptimizer
from ...status.orchestrator_status import OrchestratorStatus


class FedProxOptimizer(GlobalUpdateOptimizer):

    def __init__(self, epochs: int, batch_size: int, num_examples: int, status: OrchestratorStatus, logger):
        super().__init__(epochs, batch_size, num_examples, status, logger)
        self.p_heterogeneity = 0.5

    def optimize(self, r: int, dev_index: int, phase: str) -> dict:
        is_heterogeneous = np.random.binomial(1, self.p_heterogeneity)
        if not is_heterogeneous or self.epochs == 0:
            return {"epochs": self.epochs, "batch_size": self.batch_size, "num_examples": self.num_examples}
        else:
            return {"epochs": random.randint(1, self.epochs), "batch_size": self.batch_size, "num_examples": self.num_examples}

