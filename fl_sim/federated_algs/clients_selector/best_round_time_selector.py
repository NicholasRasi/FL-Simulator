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

            computation_times = self.get_computation_times(num_round)
            communication_times = self.get_network_consumption(num_round) / self.status.con["devs"]["net_speed"][num_round]

            # Compute total times
            times = computation_times + communication_times

            # Select fastest
            dev_indexes = sorted(range(len(times)), key=lambda x: times[x])[:num_devs]

        return dev_indexes
