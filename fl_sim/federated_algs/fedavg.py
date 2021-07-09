import statistics as stats
from json_tricks import dump
import numpy as np
import os
from tqdm import trange
from fl_sim import Status
from fl_sim.configuration import Config
from fl_sim.aggregation_strategy import FedAvgAgg
from fl_sim.clients_selector.clients_selector_factory import ClientsSelectorFactory
from fl_sim.global_update_optimizer import GlobalUpdateOptimizerFactory
from fl_sim.local_data_optimizer import LocalDataOptimizerFactory
from fl_sim.status.dataset_model_loader import DatasetModelLoader



class FedAvg:

    def __init__(self, config: Config, logger):
        self.status = None
        self.clients_selector = None
        self.global_update_optimizer = None
        self.local_data_optimizer = None
        self.config = config
        self.logger = logger
        self.run_data = []
        self.output_dir = self.config.simulation["output_folder"]
        model_loader = DatasetModelLoader(config.simulation["model_name"])
        self.x_train, self.y_train, self.x_test, self.y_test = model_loader.get_dataset()

    def init_status(self):
        self.status = Status(logger=self.logger, config=self.config)

    def init_optimizers(self):
        self.clients_selector = {
            "fit": ClientsSelectorFactory.get_clients_selector(self.config.algorithms["fit"]["selection"], self.config,
                                                               self.status, self.logger),
            "eval": ClientsSelectorFactory.get_clients_selector(self.config.algorithms["eval"]["selection"], self.config,
                                                                self.status, self.logger)
        }
        self.global_update_optimizer = {
            "fit": GlobalUpdateOptimizerFactory.get_optimizer(self.config.algorithms["fit"]["update"],
                                                              self.config.algorithms["fit"]["params"]["epochs"],
                                                              self.config.algorithms["fit"]["params"]["batch_size"],
                                                              self.config.algorithms["fit"]["params"]["num_examples"],
                                                              self.status, self.logger),
            "eval": GlobalUpdateOptimizerFactory.get_optimizer(self.config.algorithms["eval"]["update"],
                                                               0,
                                                               self.config.algorithms["eval"]["params"]["batch_size"],
                                                               self.config.algorithms["eval"]["params"]["num_examples"],
                                                               self.status, self.logger)
        }
        self.local_data_optimizer = {
            "fit": LocalDataOptimizerFactory.get_optimizer(self.config.algorithms["fit"]["data"], self.status, self.logger),
            "eval": LocalDataOptimizerFactory.get_optimizer(self.config.algorithms["eval"]["data"], self.status, self.logger),
        }


    def save_run_data(self):
        self.run_data.append(self.status.to_dict())

    def export_all_data(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.output_dir + "/" + self.config.simulation["output_file"], 'w') as fp:
            dump({"status": self.run_data, "config": self.config.__dict__}, fp)

    def select_devs(self, r: int, phase: str):
        if phase == "fit":
            dev_indexes = self.clients_selector["fit"].select_devices(r)
        else:  # eval
            dev_indexes = self.clients_selector["eval"].select_devices(r)
        # update status
        self.status.var[phase]["devs"]["selected"][r, dev_indexes] = 1
        return dev_indexes

    def update_optimizer_configs(self, r: int, dev_index: int, phase: str, location: str, config: dict):
        self.status.var[phase]["upd_opt_configs"][location]["epochs"][r, dev_index] = config["epochs"]
        self.status.var[phase]["upd_opt_configs"][location]["batch_size"][r, dev_index] = config["batch_size"]
        self.status.var[phase]["upd_opt_configs"][location]["num_examples"][r, dev_index] = config["num_examples"]

    def update_agg_model_metrics(self, r: int, phase: str, agg_loss: float, agg_accuracy: float):
        self.status.var[phase]["model_metrics"]["agg_loss"][r] = agg_loss
        self.status.var[phase]["model_metrics"]["agg_accuracy"][r] = agg_accuracy

    def update_sim_data(self, r: int, phase: str, dev_index: int, computation_time: float, communication_time: float,
                        local_iterations: float, network_consumption: float, energy_consumption: float, accuracy: float,
                        loss: float):
        self.status.var[phase]["times"]["computation"][r, dev_index] = computation_time
        self.status.var[phase]["times"]["communication"][r, dev_index] = communication_time
        self.status.var[phase]["consumption"]["resources"][r, dev_index] = local_iterations
        self.status.var[phase]["consumption"]["network"][r, dev_index] = network_consumption
        self.status.var[phase]["consumption"]["energy"][r, dev_index] = energy_consumption
        self.status.var[phase]["model_metrics"]["accuracy"][r, dev_index] = accuracy
        self.status.var[phase]["model_metrics"]["loss"][r, dev_index] = loss

    def get_failed_devs(self, r: int):
        return np.where(self.status.con["devs"]["failures"][r] == 1)[0]

    def load_local_data(self, phase: str, dev_index: int):
        if phase == "fit":
            x = self.x_train[self.status.con["devs"]["local_data"][0][dev_index]]
            y = self.y_train[self.status.con["devs"]["local_data"][0][dev_index]]
        else:  # eval
            x = self.x_test[self.status.con["devs"]["local_data"][1][dev_index]]
            y = self.y_test[self.status.con["devs"]["local_data"][1][dev_index]]
        return x, y

    def run_server(self):
        for repetition in range(1, self.config.simulation["repetitions"]+1):
            self.logger.info("starting repetition {}/{}".format(repetition, self.config.simulation["repetitions"]))

            self.logger.info("init status and optimizers...")
            self.init_status()
            self.init_optimizers()

            self.logger.info("starting training...")

            for r in trange(self.config.simulation["num_rounds"]):
                self.model_fit(r)
                self.model_eval(r)
            self.logger.info("training completed")
            self.logger.info("saving run data")
            self.save_run_data()

        self.export_all_data()
        self.logger.info("export to {} completed".format(self.config.simulation["output_file"]))

    def model_fit(self, r: int):
        # local fit
        # select devices
        dev_indexes = self.select_devs(r, "fit")

        # start local fit execution
        local_fits = []
        failing_devs_indexes = self.get_failed_devs(r)
        for dev_index in dev_indexes:
            # run update optimizer
            global_config = self.global_update_optimizer["fit"].optimize(r, dev_index, "fit")

            # update global configs status
            self.update_optimizer_configs(r, dev_index, "fit", "global", global_config)

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
            else:
                # run client fit
                num_examples, weights, loss, accuracy = self.run_client_fit(r, dev_index, global_config)
                local_fits.append((num_examples, weights, loss, accuracy))

        if len(local_fits) > 0:  # at least on successful client
            weights = [(r[0], r[1]) for r in local_fits]
            losses = [(r[0], r[2]) for r in local_fits]
            accuracies = [(r[0], r[3]) for r in local_fits]

            # aggregate local results
            aggregated_weights = FedAvgAgg.aggregate_fit(weights)
            aggregated_loss = FedAvgAgg.aggregate_losses(losses)
            aggregated_accuracy = FedAvgAgg.aggregate_accuracies(accuracies)

            # update global model and model metrics
            self.status.global_model.set_weights(aggregated_weights)
            self.update_agg_model_metrics(r, "fit", aggregated_loss, aggregated_accuracy)
        else:
            self.logger.error("round failed")

    def model_eval(self, r: int):
        # local evaluation
        # select devices
        dev_indexes = self.select_devs(r, "eval")

        # start local eval execution
        local_evals = []
        failing_devs_indexes = self.get_failed_devs(r)
        for dev_index in dev_indexes:
            # run global update optimizer
            global_config = self.global_update_optimizer["eval"].optimize(r, dev_index, "eval")

            # update global configs status
            self.update_optimizer_configs(r, dev_index, "eval", "global", global_config)

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
            else:
                # run client eval
                num_examples, loss, accuracy = self.run_client_eval(r, dev_index, global_config)
                local_evals.append((num_examples, loss, accuracy))

        if len(local_evals) > 0:  # at least on successful client
            losses = [(r[0], r[1]) for r in local_evals]
            accuracies = [(r[0], r[2]) for r in local_evals]

            # aggregate local results
            aggregated_loss = FedAvgAgg.aggregate_losses(losses)
            aggregated_accuracy = FedAvgAgg.aggregate_accuracies(accuracies)

            # update model metrics
            self.update_agg_model_metrics(r, "eval", aggregated_loss, aggregated_accuracy)
        else:
            self.logger.error("round failed")

    def run_client_fit(self, r: int, dev_index: int, config: dict):
        # load training data
        x_train, y_train = self.load_local_data("fit", dev_index)

        # run local data optimizer
        x_train_sub, y_train_sub = self.local_data_optimizer["fit"].optimize(r, dev_index,
                                                                             config["num_examples"], (x_train, y_train))
        config["num_examples"] = x_train_sub.shape[0]

        # update local model
        model = self.status.con["devs"]["local_models"][dev_index]
        model.set_weights(self.status.global_model.get_weights())
        history = model.fit(x_train_sub, y_train_sub,
                            epochs=config["epochs"],
                            batch_size=config["batch_size"],
                            verbose=self.config.simulation["tf_verbosity"])
        mean_acc = stats.mean(history.history['accuracy'])
        mean_loss = stats.mean(history.history['loss'])

        # compute metrics
        local_iterations = config["epochs"] * x_train_sub.shape[0] / config["batch_size"]
        computation_time = local_iterations / self.status.con["devs"]["ips"][dev_index]
        network_consumption = 2 * self.status.con["model"]["tot_weights"]
        communication_time = 2 * self.status.con["model"]["tot_weights"] / self.status.con["devs"]["net_speed"][r, dev_index]
        energy_consumption = self.config.energy["pow_comp_s"] * computation_time + self.config.energy["pow_net_s"] * communication_time

        # update global configs status
        self.update_optimizer_configs(r, dev_index, "fit", "local", config)

        # update status
        self.update_sim_data(r, "fit", dev_index,
                             computation_time=computation_time,
                             communication_time=communication_time,
                             local_iterations=local_iterations,
                             network_consumption=network_consumption,
                             energy_consumption=energy_consumption,
                             accuracy=mean_acc,
                             loss=mean_loss)

        return x_train_sub.shape[0], model.get_weights(), mean_loss, mean_acc


    def run_client_eval(self, r: int, dev_index: int, config: dict):
        # load test data
        x_test, y_test = self.load_local_data("eval", dev_index)

        # run local data optimizer
        x_test_sub, y_test_sub = self.local_data_optimizer["eval"].optimize(r, dev_index,
                                                                            config["num_examples"], (x_test, y_test))
        config["num_examples"] = x_test_sub.shape[0]

        # evaluate model
        model = self.status.con["devs"]["local_models"][dev_index]
        model.set_weights(self.status.global_model.get_weights())
        loss, accuracy = model.evaluate(x_test_sub, y_test_sub, verbose=self.config.simulation["tf_verbosity"])

        # compute metrics
        local_iterations = x_test_sub.shape[0] / config["batch_size"]
        computation_time = local_iterations / self.status.con["devs"]["ips"][dev_index]
        network_consumption = self.status.con["model"]["tot_weights"]
        communication_time = self.status.con["model"]["tot_weights"] / self.status.con["devs"]["net_speed"][r, dev_index]
        energy_consumption = self.config.energy["pow_comp_s"] * computation_time + self.config.energy["pow_net_s"] * communication_time

        # update global configs status
        self.update_optimizer_configs(r, dev_index, "eval", "local", config)

        # update status
        self.update_sim_data(r, "eval", dev_index,
                             computation_time=computation_time,
                             communication_time=communication_time,
                             local_iterations=local_iterations,
                             network_consumption=network_consumption,
                             energy_consumption=energy_consumption,
                             accuracy=accuracy,
                             loss=loss)

        return x_test_sub.shape[0], loss, accuracy

