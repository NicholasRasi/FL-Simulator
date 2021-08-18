import unittest
from fl_sim.configuration.config import Config
import os


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_dict_merge(self):
        config_1 = {"a": 1, "b": {"b1": 2, "b2": 3, "b3": {"b31": 4}}, "c": {"c1": 5}, "d": {"d1": 6}}
        config_2 = {"b": {"b2": 30, "b3": {"b31": 40}}, "d": {"d1": 60}}
        exp_config = {"a": 1, "b": {"b1": 2, "b2": 30, "b3": {"b31": 40}}, "c": {"c1": 5}, "d": {"d1": 60}}
        self.assertDictEqual(Config._merge_dictionaries(config_1, config_2), exp_config, "Test Dict Merge")

    def test_read_config_import(self):
        config_file_1 = os.path.join(os.path.dirname(__file__), "config_1.yml")
        config_file_2 = os.path.join(os.path.dirname(__file__), "result_config.yml")
        config_1 = Config(config_file=config_file_1)
        config_2 = Config(config_file=config_file_2)
        self.assertDictEqual(config_1.config, config_2.config, "Test Read Config Import")


if __name__ == '__main__':
    unittest.main()
