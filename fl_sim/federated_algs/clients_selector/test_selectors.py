import unittest
from fl_sim.federated_algs.clients_selector.best_ips_selector import BestIPSSelector
import numpy as np


class TestSelectors(unittest.TestCase):

    def test_best_ips(self):
        devs_ips = np.array([6, 1, 2, 3, 4, 5])
        avail_indexes = np.array([0, 1, 2, 4, 5])
        num_devs = 4
        selection = BestIPSSelector.select_best_ips(devs_ips, avail_indexes, num_devs)
        self.assertEqual(selection, [0, 5, 4, 2], "Test BestIPS strategy")


if __name__ == '__main__':
    unittest.main()
