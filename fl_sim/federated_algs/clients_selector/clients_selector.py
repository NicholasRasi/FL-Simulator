from abc import abstractmethod, ABC
from typing import List
from fl_sim.configuration import Config
from ...status.orchestrator_status import OrchestratorStatus
import numpy as np


class ClientsSelector(ABC):

    def __init__(self, config: Config, status: OrchestratorStatus, logger, params=None):
        self.config = config
        self.status = status
        self.logger = logger
        self.params = params

    @abstractmethod
    def select_devices(self, num_round: int) -> List:
        """
        Select devices
        :param num_round: the devices index selected for the current round
        """

    def get_available_devices(self, num_round):
        avail_devs = self.status.con["devs"]["availability"][num_round]
        avail_indexes = np.where(avail_devs == 1)[0]
        return avail_indexes

    def get_quadratic_mean_times(self, num_round):
        comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
        comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])

        global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
        current_local_iterations = global_opt_configs["epochs"][num_round] * \
                                   global_opt_configs["num_examples"][
                                       num_round] / global_opt_configs["batch_size"][num_round]
        expected_computation_times = np.divide(current_local_iterations, self.status.con["devs"]["ips"])
        comm_times = comm_upload_times + comm_distribution_times
        mean_square_times = np.asarray(
            [0 if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) + comp for t, comp in
             zip(comm_times, expected_computation_times)])
        return mean_square_times

    def get_quadratic_mean_computation_times(self, num_round):
        global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
        current_local_iterations = global_opt_configs["epochs"][num_round] * global_opt_configs["num_examples"][
            num_round] / global_opt_configs["batch_size"][num_round]
        expected_computation_times = np.divide(current_local_iterations, self.status.con["devs"]["ips"])
        return expected_computation_times

    def get_quadratic_mean_communication_times(self):
        comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
        comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])
        comm_times = comm_upload_times + comm_distribution_times
        mean_square_comm_times = np.asarray(
            [0 if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) for t in comm_times])
        return mean_square_comm_times
