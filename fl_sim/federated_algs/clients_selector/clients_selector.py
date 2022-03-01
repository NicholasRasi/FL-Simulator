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

    def get_computation_times(self, num_round):
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

    def get_resources_consumption(self, num_round):
        global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
        resources_consumption = global_opt_configs["epochs"][num_round] * global_opt_configs["num_examples"][
            num_round] / global_opt_configs["batch_size"][num_round]
        return resources_consumption

    def get_network_consumption(self, num_round):
        network_consumption = np.amax(self.status.var["fit"]["consumption"]["network_upload"]) + np.amax(self.status.var["fit"]["consumption"]["network_distribution"])
        return network_consumption

    def get_estimated_energy_consumption(self, num_round):
        estimated_energy_consumption = self.config.energy["pow_comp_s"] * self.get_computation_times(num_round) + self.config.energy[
            "pow_net_s"] * self.get_quadratic_mean_communication_times()
        return estimated_energy_consumption

    def get_energy_consumption(self, num_round):
        communication_times = self.get_network_consumption(num_round) / self.status.con["devs"]["net_speed"][num_round]
        energy_consumption = self.config.energy["pow_comp_s"] * self.get_computation_times(num_round) + \
                             self.config.energy["pow_net_s"] * communication_times
        return energy_consumption




