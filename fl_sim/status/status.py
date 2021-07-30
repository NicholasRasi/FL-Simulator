import numpy as np
import sys
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader import DatasetModelLoader
from fl_sim.utils import FedPhase


class Status:

    def __init__(self, config: Config, logger):
        self.logger = logger
        self.devices = {}

        if "random_seed" in config.simulation:
            np.random.seed(config.simulation["random_seed"])

        if config.simulation["initializer"] == "default":
            # CONSTANT DATA
            # init dataset
            x_train, y_train, x_test, y_test = DatasetModelLoader(config.simulation["model_name"]).get_dataset()

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
                    "local_models_weights": [None] * config.devices["num"]
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
            self.global_model_weights = None

            # number of parameters of the model (initialized after the first train)
            self.con["model"] = {
                "tot_weights": None
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
                        "metric": np.zeros(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                             dtype=float),
                        "loss": np.full(shape=(config.simulation["num_rounds"], config.devices["num"]),
                                        fill_value=sys.float_info.max, dtype=float),
                        "agg_metric": np.zeros(shape=config.simulation["num_rounds"], dtype=float),
                        "agg_loss": np.full(shape=config.simulation["num_rounds"],
                                            fill_value=sys.float_info.max, dtype=float)
                    }}
                for phase in ["fit", "eval"]}


    def update_optimizer_configs(self, num_round: int, dev_index: int, fed_phase: FedPhase, location: str, config: dict):
        phase = fed_phase.value
        self.var[phase]["upd_opt_configs"][location]["epochs"][num_round, dev_index] = config["epochs"]
        self.var[phase]["upd_opt_configs"][location]["batch_size"][num_round, dev_index] = config["batch_size"]
        self.var[phase]["upd_opt_configs"][location]["num_examples"][num_round, dev_index] = config["num_examples"]

    def update_agg_model_metrics(self, num_round: int, fed_phase: FedPhase, agg_loss: float, agg_metric: float):
        phase = fed_phase.value
        self.var[phase]["model_metrics"]["agg_loss"][num_round] = agg_loss
        self.var[phase]["model_metrics"]["agg_metric"][num_round] = agg_metric

    def update_sim_data(self, num_round: int, fed_phase: FedPhase, dev_index: int, computation_time: float,
                        communication_time: float, local_iterations: float, network_consumption: float,
                        energy_consumption: float, metric: float, loss: float):
        phase = fed_phase.value
        self.var[phase]["times"]["computation"][num_round, dev_index] = computation_time
        self.var[phase]["times"]["communication"][num_round, dev_index] = communication_time
        self.var[phase]["consumption"]["resources"][num_round, dev_index] = local_iterations
        self.var[phase]["consumption"]["network"][num_round, dev_index] = network_consumption
        self.var[phase]["consumption"]["energy"][num_round, dev_index] = energy_consumption
        self.var[phase]["model_metrics"]["metric"][num_round, dev_index] = metric
        self.var[phase]["model_metrics"]["loss"][num_round, dev_index] = loss


    def resize_status(self, num_rounds):
        self.con["devs"]["availability"] = self.con["devs"]["availability"][:num_rounds, :]
        self.con["devs"]["failures"] = self.con["devs"]["failures"][:num_rounds, :]
        self.con["devs"]["energy"] = self.con["devs"]["energy"][:num_rounds, :]
        self.con["devs"]["net_speed"] = self.con["devs"]["net_speed"][:num_rounds, :]

        for phase in ["fit", "eval"]:
            self.var[phase]["devs"]["selected"] = self.var[phase]["devs"]["selected"][:num_rounds, :]
            for loc in ["global", "local"]:
                for config in ["epochs", "batch_size", "num_examples"]:
                    self.var[phase]["upd_opt_configs"][loc][config] = self.var[phase]["upd_opt_configs"][loc][config][:num_rounds, :]
            for time in ["computation", "communication"]:
                self.var[phase]["times"][time] = self.var[phase]["times"][time][:num_rounds, :]
            for consumption in ["resources", "network", "energy"]:
                self.var[phase]["consumption"][consumption] = self.var[phase]["consumption"][consumption][:num_rounds, :]
            for metric in ["metric", "loss"]:
                self.var[phase]["model_metrics"][metric] = self.var[phase]["model_metrics"][metric][:num_rounds, :]
            for metric in ["agg_metric", "agg_loss"]:
                self.var[phase]["model_metrics"][metric] = self.var[phase]["model_metrics"][metric][:num_rounds]


    @staticmethod
    def randint(mean: int, var: int, size, dtype):
        if var == 0:
            return np.full(shape=size, fill_value=mean, dtype=dtype)
        else:
            return np.random.randint(low=mean-var, high=mean+var, size=size, dtype=dtype)

    def to_dict(self):
        self.con["devs"]["local_data"] = None
        self.con["devs"]["local_models_weights"] = None
        return {"con": self.con,
                "var": self.var}
