import sys
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class LimitedTimeSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)
        self.computation_time_limit = None
        self.communication_time_limit = None
        self.total_time_limit = None
        self.auto_tuning = False

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # If it's the first round or there are no time limitations then extract randomly num_devs devices
        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        # Otherwise
        else:

            # 1. Filter devices depending on time limitations
            in_time_devices = set(avail_indexes)

            if self.auto_tuning:
                self.apply_autotuning(num_round)

            if self.communication_time_limit is not None:
                mean_square_comm_times = self.get_quadratic_mean_communication_times()
                out_of_time_devices = set(np.where(mean_square_comm_times > self.communication_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            if self.computation_time_limit is not None:
                expected_computation_times = self.get_computation_times(num_round)
                out_of_time_devices = set(np.where(expected_computation_times > self.computation_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            if self.total_time_limit is not None:
                mean_square_times = self.get_quadratic_mean_times(num_round)
                out_of_time_devices = set(np.where(mean_square_times > self.total_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices

            # 2. Then choose num_devs devices with biggest loss
            losses = [sys.float_info.max if len(x[x < sys.float_info.max]) == 0 else x[
                np.where(x != sys.float_info.max)[0][-1]] for x in
                      np.transpose(self.status.var["fit"]["model_metrics"]["loss"])]
            biggest_loss = [x for x in np.argsort(losses) if x in in_time_devices]
            dev_indexes = biggest_loss[-num_devs:]

        return dev_indexes

    # Exclude devices with time "too far" from mean time
    def apply_autotuning(self, num_round):

        # Compute average round time of all devices
        mean_square_times = self.get_quadratic_mean_times(num_round)
        mean_time = np.mean(mean_square_times)

        # Find differences between average round time of all devices and mean round time of each device
        deviations = [x - mean_time for x in mean_square_times]

        # Compute mean absolute deviation
        mean_abs_deviation = np.mean(np.abs(deviations))

        # Set total limit as the average round time of all devices + the average distance
        self.total_time_limit = mean_time + mean_abs_deviation
