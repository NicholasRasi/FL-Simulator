from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector


class BestIPSSelector(ClientsSelector):

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # sort devices based on IPS
        return BestIPSSelector.select_best_ips(self.status.con["devs"]["ips"], avail_indexes, num_devs)

    @staticmethod
    def select_best_ips(devs_ips, avail_indexes, num_devs):
        sorted_devs_ips = np.argsort(devs_ips)[::-1]
        dev_indexes = []
        for d in sorted_devs_ips:
            if d in avail_indexes:
                dev_indexes.append(d)
            if len(dev_indexes) == num_devs:
                return dev_indexes
        return dev_indexes
