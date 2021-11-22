import math
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class DynamicSamplingSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)
        self.sampling_decay_coefficient = 0.1

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        C = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])
        c = int(C / math.exp(num_round*self.sampling_decay_coefficient))
        dev_indexes = np.random.choice(avail_indexes, size=max(c, 1), replace=False)
        return dev_indexes
