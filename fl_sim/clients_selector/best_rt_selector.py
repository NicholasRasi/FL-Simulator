from typing import List
import numpy as np
from fl_sim.clients_selector.clients_selector import ClientsSelector


class BestRTSelector(ClientsSelector):

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.k_fit * avail_indexes.shape[0])

        # sort devices based on IPS
        sorted_devs_ips = np.argsort(self.status.con["devs"]["ips"])[::-1]
        dev_indexes = []
        for d in sorted_devs_ips:
            if d in avail_indexes:
                dev_indexes.append(d)
            if len(dev_indexes) == num_devs:
                return dev_indexes
        return dev_indexes
