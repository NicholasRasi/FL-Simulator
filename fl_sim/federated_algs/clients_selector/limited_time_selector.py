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
        self.total_time_limit = 8

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # If it's the first round or there is no time limitations then extract randomly num_devs devices
        if num_round == 0 or (self.communication_time_limit == self.computation_time_limit == self.total_time_limit is None):
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:
            in_time_devices = set(avail_indexes)
            if self.communication_time_limit is not None:
                comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
                comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])
                comm_times = comm_upload_times + comm_distribution_times
                mean_square_comm_times = np.asarray([sys.float_info.max if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) for t in comm_times])
                out_of_time_devices = set(np.where(mean_square_comm_times > self.communication_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            if self.computation_time_limit is not None:
                global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
                current_local_iterations = global_opt_configs["epochs"][num_round] * global_opt_configs["num_examples"][
                    num_round] / global_opt_configs["batch_size"][num_round]
                expected_computation_times = np.divide(current_local_iterations, self.status.con["devs"]["ips"])
                out_of_time_devices = set(np.where(expected_computation_times > self.computation_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            if self.total_time_limit is not None:
                comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
                comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])
                comm_times = comm_upload_times + comm_distribution_times
                mean_square_comm_times = np.asarray([sys.float_info.max if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) for t in comm_times])
                global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
                current_local_iterations = global_opt_configs["epochs"][num_round] * global_opt_configs["num_examples"][
                    num_round] / global_opt_configs["batch_size"][num_round]
                expected_computation_times = np.divide(current_local_iterations, self.status.con["devs"]["ips"])
                total_times = np.asarray([comm + comp for comm, comp in zip(mean_square_comm_times, expected_computation_times)])
                out_of_time_devices = set(np.where(total_times > self.total_time_limit)[0])
                in_time_devices = in_time_devices - out_of_time_devices
            dev_indexes = np.random.choice(list(in_time_devices), size=min(num_devs, len(in_time_devices)), replace=False)

        return dev_indexes