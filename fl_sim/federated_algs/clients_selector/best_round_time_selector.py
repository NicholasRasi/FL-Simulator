from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector


class BestRoundTimeSelector(ClientsSelector):

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        if num_round == 0:
            # If first round there is no history data or information about model weights size then select randomly
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:
            # Otherwise compute the expected time for each device and select the fastest devices

            # Compute computation time
            global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
            local_iterations = global_opt_configs["epochs"][num_round] * global_opt_configs["num_examples"][num_round] / global_opt_configs["batch_size"][num_round]
            computation_times = local_iterations / self.status.con["devs"]["ips"]

            # Compute communication time
            network_consumption_upload = self.status.var["fit"]["consumption"]["network_upload"][num_round - 1]
            network_parameters_upload = network_consumption_upload[network_consumption_upload > 0][0]
            network_consumption_distribution = self.status.var["fit"]["consumption"]["network_distribution"][num_round - 1]
            network_parameters_distribution = network_consumption_distribution[network_consumption_distribution > 0][0]
            upload_times = network_parameters_upload / self.status.con["devs"]["net_speed"][num_round]
            distribution_times = network_parameters_distribution / self.status.con["devs"]["net_speed"][num_round]

            # Compute total times
            times = computation_times + upload_times + distribution_times

            # Select fastest
            dev_indexes = sorted(range(len(times)), key=lambda x: times[x])[:num_devs]

        return dev_indexes
