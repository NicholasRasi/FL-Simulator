import unittest
from fl_sim.federated_algs.clients_selector.best_selector import BestSelector
import numpy as np


class TestSelectors(unittest.TestCase):

    def test_best_ips(self):
        devs_cap = np.array([6, 1, 2, 3, 4, 5])
        avail_indexes = np.array([0, 1, 2, 4, 5])
        num_devs = 4
        selection = BestSelector.select_best(devs_cap, avail_indexes, num_devs)
        self.assertEqual(selection, [0, 5, 4, 2], "Test Best strategy")


if __name__ == '__main__':
    unittest.main()
