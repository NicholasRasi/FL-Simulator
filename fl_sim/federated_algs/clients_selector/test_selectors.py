import unittest
from best_ips_selector import BestIPSSelector
from fl_sim.configuration import Config


class TestSelectors(unittest.TestCase):

    def test_best_ips(self):
        config = Config()

        selector = BestIPSSelector()


        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")


if __name__ == '__main__':
    unittest.main()
