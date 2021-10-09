import numpy as np
import sys
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from fl_sim.utils import FedPhase


class WorkerStatus:

    def __init__(self, config: Config, logger):
        self.logger = logger
        self.devices = {}

        if "random_seed" in config.simulation:
            np.random.seed(config.simulation["random_seed"])

        if config.simulation["initializer"] == "default":
            # CONSTANT DATA
            # init dataset
            x_train, y_train, x_test, y_test = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"]).get_dataset()

            # init constant simulation data
            self.con = {
                "devs": {
                    "local_data_sizes": self.randint(mean=config.data["num_examples_mean"],
                                                     var=config.data["num_examples_var"],
                                                     size=config.devices["num"], dtype=int),
                    "local_data": (),
                    "local_data_stats": None,
                    "local_models_weights": [None] * config.devices["num"]
                }
            }

            # init local data
            if config.data["non_iid_partitions"] > 0:
                # non-iid partitions
                train_indexes = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"]).select_non_iid_samples(y_train, config.devices["num"],
                                                                          self.con["devs"]["local_data_sizes"],
                                                                          config.data["non_iid_partitions"])
                eval_indexes = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"]).select_random_samples(y_test, config.devices["num"],
                                                                        self.con["devs"]["local_data_sizes"])
            else:
                # random sampling
                train_indexes = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"]).select_random_samples(y_train, config.devices["num"],
                                                                         self.con["devs"]["local_data_sizes"])
                eval_indexes = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"]).select_random_samples(y_test, config.devices["num"],
                                                                        self.con["devs"]["local_data_sizes"])
            self.con["devs"]["local_data"] = (train_indexes, eval_indexes)
            self.con["devs"]["local_data_stats"] = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"]).record_data_stats(y_train, train_indexes)

    @staticmethod
    def randint(mean: int, var: int, size, dtype):
        if var == 0:
            return np.full(shape=size, fill_value=mean, dtype=dtype)
        else:
            return np.random.randint(low=mean-var, high=mean+var, size=size, dtype=dtype)

    def to_dict(self):
        self.con["devs"]["local_data"] = None
        self.con["devs"]["local_models_weights"] = None
        return {"con": self.con}
