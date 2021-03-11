import numpy as np

from config import Config
from .dataset_model_loader import DatasetModelLoader
import tensorflow.keras.backend as keras


class Status:

    def __init__(self, config: Config, logger, initializer="default"):
        self.logger = logger
        self.devices = {}

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        if initializer == "default":
            # init constant data
            # init model and dataset
            model_loader = DatasetModelLoader(config.model_name)
            x_train, y_train, x_test, y_test = model_loader.get_dataset()

            # init constant simulation data
            self.con = {
                "devs": {
                    "availability": np.random.binomial(n=1, p=config.p_available, size=(config.num_rounds, config.num_devs)),
                    "failures": np.random.binomial(n=1, p=config.p_fail, size=(config.num_rounds, config.num_devs)),
                    "ips": self.randint(mean=config.ips_mean, var=config.ips_var, size=config.num_devs, dtype=int),
                    "energy": self.randint(mean=config.energy_mean, var=config.energy_var, size=(config.num_rounds, config.num_devs), dtype=int),
                    "net_speed": self.randint(mean=config.netspeed_mean, var=config.netspeed_var, size=(config.num_rounds, config.num_devs), dtype=int),
                    "local_data_sizes": self.randint(mean=config.local_data_mean, var=config.local_data_var, size=config.num_devs, dtype=int),
                    "local_data": [],
                    "local_models": [model_loader.get_compiled_model(optimizer=config.optimizer)] * config.num_devs
                }
            }

            for local_data_size in self.con["devs"]["local_data_sizes"]:
                x_train_local, y_train_local = DatasetModelLoader.select_random_samples(x_train, y_train, local_data_size)
                x_test_local, y_test_local = DatasetModelLoader.select_random_samples(x_test, y_test, local_data_size)
                self.con["devs"]["local_data"].append(((x_train_local, y_train_local), (x_test_local, y_test_local)))

            self.global_model = self.con["devs"]["local_models"][0]

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
                        "selected": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=int)
                    },
                    "upd_opt_configs":
                        {loc: {
                            "epochs": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=int),
                            "batch_size": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=int),
                            "num_examples": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=int),
                        } for loc in ["global", "local"]
                        },
                    "times": {
                        "computation": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float),
                        "communication": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float)
                    },
                    "consumption": {
                        "resources": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float),
                        "network": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float),
                        "energy": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float)
                    },
                    "model_metrics": {
                        "accuracy": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float),
                        "loss": np.zeros(shape=(config.num_rounds, config.num_devs), dtype=float),
                        "agg_accuracy": np.zeros(shape=config.num_rounds, dtype=float),
                        "agg_loss": np.zeros(shape=config.num_rounds, dtype=float)
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
