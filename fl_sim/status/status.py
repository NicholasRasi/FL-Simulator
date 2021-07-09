import numpy as np

from fl_sim.configuration import Config
from .dataset_model_loader import DatasetModelLoader
import tensorflow.keras.backend as keras


class Status:

    def __init__(self, config: Config, logger, initializer="default"):
        self.logger = logger
        self.devices = {}

        if "random_seed" in config.simulation:
            np.random.seed(config.simulation["random_seed"])

        if initializer == "default":
            # CONSTANT DATA
            # init model and dataset
            model_loader = DatasetModelLoader(config.simulation["model_name"])
            x_train, y_train, x_test, y_test = model_loader.get_dataset()

            # init constant simulation data
            self.con = {
                "devs": {
                    "availability": np.random.binomial(n=1,
                                                       p=config.devices["p_available"],
                                                       size=(config.simulation["num_rounds"],
                                                             config.devices["num"])),
                    "failures": np.random.binomial(n=1,
                                                   p=config.devices["p_fail"],
                                                   size=(config.simulation["num_rounds"],
                                                         config.devices["num"])),
                    "ips": self.randint(mean=config.computation["ips_mean"],
                                        var=config.computation["ips_var"],
                                        size=config.devices["num"],
                                        dtype=int),
                    "energy": self.randint(mean=config.energy["avail_mean"],
                                           var=config.energy["avail_var"],
                                           size=(config.simulation["num_rounds"], config.devices["num"]),
                                           dtype=int),
                    "net_speed": self.randint(mean=config.network["speed_mean"],
                                              var=config.network["speed_var"],
                                              size=(config.simulation["num_rounds"], config.devices["num"]),
                                              dtype=int),
                    "local_data_sizes": self.randint(mean=config.data["num_examples_mean"],
                                                     var=config.data["num_examples_var"],
                                                     size=config.devices["num"], dtype=int),
                    "local_data": (),
                    "local_data_stats": None,
                    "local_models": [model_loader.get_compiled_model(optimizer=config.algorithms["optimizer"])] * config.devices["num"]
                }
            }

            # init local data
            if config.data["non_iid_partitions"] > 0:
                # non-iid partitions
                train_indexes = DatasetModelLoader.select_non_iid_samples(y_train, config.devices["num"],
                                                                          self.con["devs"]["local_data_sizes"],
                                                                          config.data["non_iid_partitions"])
                eval_indexes = DatasetModelLoader.select_random_samples(y_test, config.devices["num"],
                                                                        self.con["devs"]["local_data_sizes"])
            else:
                # random sampling
                train_indexes = DatasetModelLoader.select_random_samples(y_train, config.devices["num"],
                                                                         self.con["devs"]["local_data_sizes"])
                eval_indexes = DatasetModelLoader.select_random_samples(y_test, config.devices["num"],
                                                                        self.con["devs"]["local_data_sizes"])
            self.con["devs"]["local_data"] = (train_indexes, eval_indexes)
            self.con["devs"]["local_data_stats"] = DatasetModelLoader.record_data_stats(y_train, train_indexes)

            # init global model
            self.global_model = model_loader.get_compiled_model(optimizer=config.algorithms["optimizer"])

            # init number of trainable model weights
            trainable_count = np.sum([keras.count_params(w) for w in self.global_model.trainable_weights])
            non_trainable_count = np.sum([keras.count_params(w) for w in self.global_model.non_trainable_weights])

            self.con["model"] = {
                "trainable_weights": int(trainable_count),
                "non_trainable_weights": int(non_trainable_count),
                "tot_weights": int(trainable_count + non_trainable_count)
            }

            # init variable data and metrics
            self.var = {
                phase: {
                    "devs": {
                        "selected": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                             dtype=int)
                    },
                    "upd_opt_configs":
                        {loc: {
                            "epochs": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                               dtype=int),
                            "batch_size": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                                   dtype=int),
                            "num_examples": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                                     dtype=int),
                        } for loc in ["global", "local"]
                        },
                    "times": {
                        "computation": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                                dtype=float),
                        "communication": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                                  dtype=float)
                    },
                    "consumption": {
                        "resources": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                              dtype=float),
                        "network": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                            dtype=float),
                        "energy": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                           dtype=float)
                    },
                    "model_metrics": {
                        "accuracy": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                             dtype=float),
                        "loss": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                         dtype=float),
                        "agg_accuracy": np.zeros(shape=config.simulation["num_rounds"], dtype=float),
                        "agg_loss": np.zeros(shape=config.simulation["num_rounds"], dtype=float)
                    }}
                for phase in ["fit", "eval"]}

    @staticmethod
    def randint(mean: int, var: int, size, dtype):
        if var == 0:
            return np.full(shape=size, fill_value=mean, dtype=dtype)
        else:
            return np.random.randint(low=mean-var, high=mean+var, size=size, dtype=dtype)

    def to_dict(self):
        self.con["devs"]["local_data"] = None
        self.con["devs"]["local_models"] = None
        return {"con": self.con,
                "var": self.var}
