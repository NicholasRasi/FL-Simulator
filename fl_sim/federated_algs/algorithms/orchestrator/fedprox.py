from fl_sim import FedAvg
from fl_sim.configuration import Config
from fl_sim.federated_algs.global_update_optimizer.uniform_optimizer import UniformOptimizer
from fl_sim.status.orchestrator_status import OrchestratorStatus


class FedProx(FedAvg):

    def __init__(self, status: OrchestratorStatus, config: Config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock):
        super().__init__(status, config, logger,  jobs_queue, completed_jobs_queue, workers_queue, lock)
