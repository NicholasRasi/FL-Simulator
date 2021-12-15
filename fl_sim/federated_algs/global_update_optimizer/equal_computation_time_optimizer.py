from .global_update_optimizer import GlobalUpdateOptimizer
from ...status.orchestrator_status import OrchestratorStatus


class EqualComputationTimeOptimizer(GlobalUpdateOptimizer):

    # Assumption: ips of devices are known
    def __init__(self, epochs: int, batch_size: int, num_examples: int, status: OrchestratorStatus, logger):
        super().__init__(epochs, batch_size, num_examples, status, logger)
        self.computation_time = 0.15
        self.is_epochs_varying = True
        self.is_num_examples_varying = False
        self.is_batch_size_varying = False

    def optimize(self, r: int, dev_index: int, phase: str) -> dict:

        ips = self.status.con["devs"]["ips"][dev_index]
        local_iterations = self.computation_time * ips

        if self.is_epochs_varying:
            self.epochs = int(local_iterations * self.batch_size / self.num_examples)
        elif self.is_batch_size_varying:
            self.batch_size = int(self.epochs * self.num_examples / local_iterations)
        elif self.is_num_examples_varying:
            self.num_examples = int(local_iterations * self.batch_size / self.epochs)

        return {"epochs": self.epochs, "batch_size": self.batch_size, "num_examples": self.num_examples}

