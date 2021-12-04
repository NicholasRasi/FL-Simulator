from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class LimitedTimeSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)
        self.computation_time_limit = None
        self.communication_time_limit = None
        self.total_time_limit = 8

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        if num_round == 0 or (self.communication_time_limit == self.computation_time_limit == self.total_time_limit is None):
            # If it's the first round or there are no time limitations then extract randomly num_devs devices
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:
            # Otherwise

            # 1. Filter devices depending on time limitations
            in_time_devices = set(avail_indexes)
            if self.communication_time_limit is not None:
                mean_square_comm_times = self.get_quadratic_mean_communication_times()
                out_of_time_devices = set(np.where(mean_square_comm_times > self.communication_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            if self.computation_time_limit is not None:
                expected_computation_times = self.get_quadratic_mean_computation_times(num_round)
                out_of_time_devices = set(np.where(expected_computation_times > self.computation_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            if self.total_time_limit is not None:
                mean_square_times = self.get_quadratic_mean_times(num_round)
                out_of_time_devices = set(np.where(mean_square_times > self.total_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices

            # 2. Then choose num_devs devices randomly among them
            dev_indexes = np.random.choice(list(in_time_devices), size=min(num_devs, len(in_time_devices)), replace=False)

        return dev_indexes