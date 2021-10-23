from fl_sim import FedAvg
from fl_sim.configuration import Config
from fl_sim.federated_algs.global_update_optimizer.fedprox_optimizer import FedProxOptimizer
from fl_sim.status.orchestrator_status import OrchestratorStatus


class FedProx(FedAvg):

    def __init__(self, status: OrchestratorStatus, config: Config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock):
        super().__init__(status, config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock)

        self.global_update_optimizer = {
            "fit": FedProxOptimizer(self.config.algorithms["fit"]["params"]["epochs"], self.config.algorithms["fit"]["params"]["batch_size"],
                                                              self.config.algorithms["fit"]["params"]["num_examples"],
                                                              self.status, self.logger),
            "eval": FedProxOptimizer(0, self.config.algorithms["eval"]["params"]["batch_size"],
                                                               self.config.algorithms["eval"]["params"]["num_examples"],
                                                               self.status, self.logger)
        }
