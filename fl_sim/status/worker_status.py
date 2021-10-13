import numpy as np
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from fl_sim.federated_algs.local_data_optimizer import LocalDataOptimizerFactory


class WorkerStatus:

    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.dev_num = None
        self.fedalg = None
        self.verbosity = None
        self.optimizer = None
        self.metric = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.non_iid_partitions = None
        self.model_loader = None
        self.data_sizes = None
        self.train_indexes = None
        self.eval_indexes = None
        self.local_optimizer_fit = None
        self.local_optimizer_eval = None
        self.local_data_stats = None
        if "random_seed" in config.simulation:
            np.random.seed(self.config.simulation["random_seed"])

    def initialize_global_fields(self, json_fields):
        self.dataset = json_fields["dataset"]
        self.dev_num = json_fields["dev_num"]
        self.fedalg = json_fields["fedalg"]
        self.verbosity = json_fields["verbosity"]
        self.optimizer = json_fields["optimizer"]
        self.metric = json_fields["metric"]
        self.non_iid_partitions = self.config.data["non_iid_partitions"]
        self.model_loader = DatasetModelLoaderFactory.get_model_loader(self.dataset, self.dev_num)
        self.local_optimizer_fit = LocalDataOptimizerFactory.get_optimizer(json_fields["local_optimizer_fit"])
        self.local_optimizer_eval = LocalDataOptimizerFactory.get_optimizer(json_fields["local_optimizer_eval"])
        self.data_sizes = self.randint(mean=self.config.data["num_examples_mean"],
                                       var=self.config.data["num_examples_var"],
                                       size=self.dev_num, dtype=int)

        self.x_train, self.y_train, self.x_test, self.y_test = self.model_loader.get_dataset(self.config.data["mislabelling_percentage"])


        if self.non_iid_partitions > 0:
            # non-iid partitions
            self.train_indexes = self.model_loader.select_non_iid_samples(self.y_train, self.dev_num, self.data_sizes, self.non_iid_partitions)
            self.eval_indexes = self.model_loader.select_random_samples(self.y_test, self.dev_num, self.data_sizes)
        else:
            # random sampling
            self.train_indexes = self.model_loader.select_random_samples(self.y_train, self.dev_num, self.data_sizes)
            self.eval_indexes = self.model_loader.select_random_samples(self.y_test, self.dev_num, self.data_sizes)
        self.local_data_stats = self.model_loader.record_data_stats(self.y_train, self.train_indexes)

    @staticmethod
    def randint(mean: int, var: int, size, dtype):
        if var == 0:
            return np.full(shape=size, fill_value=mean, dtype=dtype)
        else:
            return np.random.randint(low=mean-var, high=mean+var, size=size, dtype=dtype)

