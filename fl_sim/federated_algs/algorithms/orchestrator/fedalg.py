import numpy as np
from abc import ABC
from fl_sim.configuration import Config
from fl_sim.status.orchestrator_status import OrchestratorStatus


class FedAlg(ABC):

    def __init__(self, status: OrchestratorStatus, config: Config, logger):
        self.status = status
        self.config = config
        self.logger = logger

    def get_failed_devs(self, num_round: int):
        return np.where(self.status.con["devs"]["failures"][num_round] == 1)[0]