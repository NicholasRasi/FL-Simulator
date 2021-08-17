from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector


class BestSelector(ClientsSelector):

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # sort devices based on sortBy param
        devs_caps = self.status.con["devs"][self.params["sortBy"]]
        # if devs_caps is a round x device array, select the current round
        if len(devs_caps.shape) > 1:
            devs_caps = devs_caps[num_round]
        return BestSelector.select_best(devs_caps, avail_indexes, num_devs)

    @staticmethod
    def select_best(devs_caps, avail_indexes, num_devs):
        sorted_devs_caps = np.argsort(devs_caps)[::-1]
        dev_indexes = []
        for d in sorted_devs_caps:
            if d in avail_indexes:
                dev_indexes.append(d)
            if len(dev_indexes) == num_devs:
                return dev_indexes
        return dev_indexes
