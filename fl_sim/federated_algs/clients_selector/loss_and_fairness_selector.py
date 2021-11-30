import math
import sys
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class LossAndFairnessSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)
        self.gamma_parameter = 0.2
        self.delta_parameter = 10

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:
            gamma = [math.pow(self.gamma_parameter, num_round - r) for r in range(self.config.simulation["num_rounds"])]
            T = sum(gamma)
            participation = self.status.var["fit"]["devs"]["selected"] - (self.status.var["fit"]["devs"]["selected"] & self.status.con["devs"]["failures"])
            N = np.dot(gamma, participation)
            scaled_N = [math.log(T / elem) for elem in N]
            U = [math.sqrt(2 * self.delta_parameter**2 * elem) for elem in scaled_N]
            losses = np.transpose([[y if y != sys.float_info.max else 0 for y in x] for x in self.status.var["fit"]["model_metrics"]["loss"]])
            cumulative_losses = [np.cumsum(loss) / self.config.simulation["num_rounds"] for loss in losses]
            L = [n * sum(cum_loss) for n, cum_loss in zip(N, cumulative_losses)]
            exploitation = np.divide(L, N)
            exploration = U
            UCB_indexes = exploration + exploitation
            dev_indexes = sorted(range(len(UCB_indexes)), key=lambda x: UCB_indexes[x])[-num_devs:]
        return dev_indexes
