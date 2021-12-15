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
        self.total_time_limit = 7
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
                in_time_devices = self.get_autotuned_devices(num_round, in_time_devices)
            else:
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

    # Remove devices with time "too far" from mean time
    def get_autotuned_devices(self, num_round, in_time_devices):
        mean_square_comm_times = self.get_quadratic_mean_communication_times()
        mean_comm_time = np.mean(mean_square_comm_times)

        # Compute mean absolute deviation
        comm_deviations = [x - mean_comm_time for x in mean_square_comm_times]
        mean_abs_comm_deviation = np.mean(np.abs(comm_deviations))

        # Remove devices whose difference wrt mean time is higher than the mean absolute deviation
        out_of_time_devices = set(np.where(comm_deviations > mean_abs_comm_deviation)[0])
        in_time_devices = in_time_devices - out_of_time_devices

        # Same for computations times
        expected_computation_times = self.get_computation_times(num_round)
        mean_comp_time = np.mean(expected_computation_times)
        comp_deviations = [x - mean_comp_time for x in expected_computation_times]
        mean_abs_comp_deviation = np.mean(np.abs(comp_deviations))
        out_of_time_devices = set(np.where(comp_deviations > mean_abs_comp_deviation)[0])
        in_time_devices = in_time_devices - out_of_time_devices

        # Same for total times
        mean_square_times = self.get_quadratic_mean_times(num_round)
        mean_time = np.mean(mean_square_times)
        deviations = [x - mean_time for x in mean_square_times]
        mean_abs_deviation = np.mean(np.abs(deviations))
        out_of_time_devices = set(np.where(deviations > mean_abs_deviation)[0])
        in_time_devices = in_time_devices - out_of_time_devices

        return in_time_devices