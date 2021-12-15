import numpy as np
import sys
import tensorflow
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from fl_sim.utils import FedPhase


class OrchestratorStatus:

    def __init__(self, config: Config, logger):
        self.logger = logger
        self.devices = {}
        self.config = config

        self.model_loader = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"],
                                                                  config.devices["num"])

        self.x_train, self.y_train, self.x_test, self.y_test = self.model_loader.get_dataset()

        self.initial_weights = self.model_loader.get_compiled_model(
            optimizer=tensorflow.keras.optimizers.get(self.config.algorithms["optimizer"]),
            metric=self.config.simulation["metric"], train_data=(self.x_train, self.y_train)).get_weights()

        if self.config.simulation["seed"] is not None:
            np.random.seed(self.config.simulation["seed"])

        self.initialize_constants()
        self.initialize_global_weights()
        self.initialize_variables()

        # init local data
        if self.config.data["non_iid_partitions"] > 0:
            # non-iid partitions
            self.train_indexes = self.model_loader.select_non_iid_samples(self.y_train, self.config.devices["num"],
                                                                     self.con["devs"]["local_data_sizes"],
                                                                     self.config.data["non_iid_partitions"])
            self.eval_indexes = self.model_loader.select_random_samples(self.y_test, self.config.devices["num"],
                                                                   self.con["devs"]["local_data_sizes"], FedPhase.EVAL)

        else:
            # random sampling
            self.train_indexes = self.model_loader.select_random_samples(self.y_train, self.config.devices["num"],
                                                                    self.con["devs"]["local_data_sizes"], FedPhase.FIT)
            self.eval_indexes = self.model_loader.select_random_samples(self.y_test, self.config.devices["num"],
                                                                   self.con["devs"]["local_data_sizes"], FedPhase.EVAL)

        self.con["devs"]["local_data"] = (self.train_indexes, self.eval_indexes)
        self.con["devs"]["local_data_stats"] = self.model_loader.record_data_stats(self.y_train, self.train_indexes)

    def reinitialize_status(self):

        if self.config.simulation["initializer"] == "default":
            if self.config.simulation["seed"] is not None:
                np.random.seed(self.config.simulation["seed"])

            # init constant simulation data
            self.initialize_constants()

            # init global weights
            self.initialize_global_weights()

            # Record data statistics
            self.con["devs"]["local_data_stats"] = self.model_loader.record_data_stats(self.y_train, self.train_indexes)

            # init variable data and metrics
            self.initialize_variables()

    def initialize_global_weights(self):
        self.global_model_weights = self.initial_weights

    def initialize_constants(self):
        self.con = {
            "devs": {
                "availability": np.random.binomial(n=1,
                                                   p=self.config.devices["p_available"],
                                                   size=(self.config.simulation["num_rounds"],
                                                         self.config.devices["num"])),
                "failures": np.random.binomial(n=1,
                                               p=self.config.devices["p_fail"],
                                               size=(self.config.simulation["num_rounds"],
                                                     self.config.devices["num"])),
                "ips": self.randint(mean=self.config.computation["ips_mean"],
                                    var=self.config.computation["ips_var"],
                                    size=self.config.devices["num"],
                                    dtype=int),
                "energy": self.randint(mean=self.config.energy["avail_mean"],
                                       var=self.config.energy["avail_var"],
                                       size=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                       dtype=int),
                "net_speed": self.get_heterogeneous_numbers(self.config.simulation["num_rounds"],
                                                         self.config.devices["num"],
                                                         self.config.network["speed_mean"],
                                                         self.config.network["speed_var"]),
                "local_data_sizes": self.randint(mean=self.config.data["num_examples_mean"],
                                                 var=self.config.data["num_examples_var"],
                                                 size=self.config.devices["num"], dtype=int),
                "local_data": (),
                "local_data_stats": None,
                "local_models_weights": [None] * self.config.devices["num"]
            },
            "model": {"tot_weights": None}
        }

    def initialize_variables(self):
        self.var = {
            phase: {
                "devs": {
                    "selected": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                         dtype=int)
                },
                "upd_opt_configs":
                    {loc: {
                        "epochs": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                           dtype=int),
                        "batch_size": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                               dtype=int),
                        "num_examples": np.zeros(
                            shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                            dtype=int),
                    } for loc in ["global", "local"]
                    },
                "times": {
                    "computation": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                            dtype=float),
                    "communication_upload": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                              dtype=float),
                    "communication_distribution": np.zeros(
                        shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                        dtype=float)
                },
                "consumption": {
                    "resources": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                          dtype=float),
                    "network_upload": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                        dtype=float),
                    "network_distribution": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                               dtype=float),
                    "energy": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                       dtype=float)
                },
                "model_metrics": {
                    "metric": np.zeros(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                       dtype=float),
                    "loss": np.full(shape=(self.config.simulation["num_rounds"], self.config.devices["num"]),
                                    fill_value=sys.float_info.max, dtype=float),
                    "agg_metric": np.zeros(shape=self.config.simulation["num_rounds"], dtype=float),
                    "agg_loss": np.full(shape=self.config.simulation["num_rounds"],
                                        fill_value=sys.float_info.max, dtype=float)
                }}
            for phase in ["fit", "eval"]}

    def update_optimizer_configs(self, num_round: int, dev_index: int, fed_phase: FedPhase, location: str, epochs: int, batch_size: int, num_examples: int):
        phase = fed_phase.value
        self.var[phase]["upd_opt_configs"][location]["epochs"][num_round, dev_index] = epochs
        self.var[phase]["upd_opt_configs"][location]["batch_size"][num_round, dev_index] = batch_size
        self.var[phase]["upd_opt_configs"][location]["num_examples"][num_round, dev_index] = num_examples

    def update_agg_model_metrics(self, num_round: int, fed_phase: FedPhase, agg_loss: float, agg_metric: float):
        phase = fed_phase.value
        self.var[phase]["model_metrics"]["agg_loss"][num_round] = agg_loss
        self.var[phase]["model_metrics"]["agg_metric"][num_round] = agg_metric

    def update_sim_data(self, num_round: int, fed_phase: FedPhase, dev_index: int, computation_time: float,
                        communication_time_upload: float, communication_time_distribution: float, local_iterations: float,
                        network_consumption_upload: float, network_consumption_distribution: float, energy_consumption: float,
                        metric: float, loss: float):
        phase = fed_phase.value
        self.var[phase]["times"]["computation"][num_round, dev_index] = computation_time
        self.var[phase]["times"]["communication_upload"][num_round, dev_index] = communication_time_upload
        self.var[phase]["times"]["communication_distribution"][num_round, dev_index] = communication_time_distribution
        self.var[phase]["consumption"]["resources"][num_round, dev_index] = local_iterations
        self.var[phase]["consumption"]["network_upload"][num_round, dev_index] = network_consumption_upload
        self.var[phase]["consumption"]["network_distribution"][num_round, dev_index] = network_consumption_distribution
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
            for time in ["computation", "communication_upload", "communication_distribution"]:
                self.var[phase]["times"][time] = self.var[phase]["times"][time][:num_rounds, :]
            for consumption in ["resources", "network_upload", "network_distribution", "energy"]:
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

    @staticmethod
    def get_heterogeneous_numbers(rounds, devices, mean, var):
        numbers = []
        means = np.random.randint(low=mean - var, high=mean + var, size=devices)
        var_degrees = np.random.uniform(low=0.01, high=0.99, size=devices)
        vars = var * var_degrees
        for dev in range(devices):
            x = np.random.randint(low=max(means[dev]-vars[dev], 0), high=means[dev]+vars[dev], size=rounds)
            numbers.append(x)
        numbers = np.transpose(np.array(numbers))
        return numbers

    def to_dict(self):
        self.global_model_weights = None
        self.con["devs"]["local_models_weights"] = None
        return {"con": self.con,
                "var": self.var}
