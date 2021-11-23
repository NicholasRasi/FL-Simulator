import math
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector


class FedCSSelector(ClientsSelector):

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:
            dev_indexes = []
            total_time = 0
            resources_consumption = self.status.var["fit"]["consumption"]["resources"][num_round - 1]
            local_iterations = resources_consumption[resources_consumption > 0][0]
            network_consumption = self.status.var["fit"]["consumption"]["network"][num_round - 1]
            network_parameters = network_consumption[network_consumption > 0][0]

            while len(dev_indexes) < num_devs:
                best_index = -1
                best_time = math.inf
                for index in avail_indexes:
                    computation_time = local_iterations / self.status.con["devs"]["ips"][index]
                    communication_time = network_parameters / self.status.con["devs"]["net_speed"][num_round, index]
                    time = communication_time + max(0, computation_time - total_time)
                    if time < best_time:
                        best_time = time
                        best_index = index
                avail_indexes = np.delete(avail_indexes, np.where(avail_indexes == best_index))
                dev_indexes.append(best_index)
                total_time = total_time + best_time
        return dev_indexes
