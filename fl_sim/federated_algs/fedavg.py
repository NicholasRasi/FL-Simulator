import statistics as stats
from fl_sim import Status
from fl_sim.configuration import Config
from fl_sim.federated_algs.fedalg import FedAlg
from fl_sim.federated_algs.aggregation_strategy import FedAvgAgg
from fl_sim.federated_algs.clients_selector import ClientsSelectorFactory
from fl_sim.federated_algs.global_update_optimizer import GlobalUpdateOptimizerFactory
from fl_sim.federated_algs.local_data_optimizer import LocalDataOptimizerFactory



class FedAvg(FedAlg):

    def __init__(self, status: Status, data, config: Config, logger):
        super().__init__(status, data, config, logger)

        self.clients_selector = None
        self.global_update_optimizer = None
        self.local_data_optimizer = None

        self.init_optimizers()

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

    def select_devs(self, r: int, phase: str):
        if phase == "fit":
            dev_indexes = self.clients_selector["fit"].select_devices(r)
        else:  # eval
            dev_indexes = self.clients_selector["eval"].select_devices(r)
        # update status
        self.status.var[phase]["devs"]["selected"][r, dev_indexes] = 1
        return dev_indexes

    def model_fit(self, num_round: int):
        # local fit
        # select devices
        dev_indexes = self.select_devs(num_round, "fit")

        # start local fit execution
        local_fits = []
        failing_devs_indexes = self.get_failed_devs(num_round)
        for dev_index in dev_indexes:
            # run update optimizer
            global_config = self.global_update_optimizer["fit"].optimize(num_round, dev_index, "fit")

            # update global configs status
            self.status.update_optimizer_configs(num_round, dev_index, "fit", "global", global_config)

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
            else:
                # run client fit
                num_examples, weights, loss, accuracy = self.run_client_fit(num_round, dev_index, global_config)
                local_fits.append((num_examples, weights, loss, accuracy))

        if len(local_fits) > 0:  # at least one successful client
            weights = [(r[0], r[1]) for r in local_fits]
            losses = [(r[0], r[2]) for r in local_fits]
            accuracies = [(r[0], r[3]) for r in local_fits]

            # aggregate local results
            aggregated_weights = FedAvgAgg.aggregate_fit(weights)
            aggregated_loss = FedAvgAgg.aggregate_losses(losses)
            aggregated_accuracy = FedAvgAgg.aggregate_accuracies(accuracies)

            # update global model and model metrics
            self.status.global_model.set_weights(aggregated_weights)
            self.status.update_agg_model_metrics(num_round, "fit", aggregated_loss, aggregated_accuracy)
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
            self.status.update_optimizer_configs(r, dev_index, "eval", "global", global_config)

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
            else:
                # run client eval
                num_examples, loss, accuracy = self.run_client_eval(r, dev_index, global_config)
                local_evals.append((num_examples, loss, accuracy))

        if len(local_evals) > 0:  # at least one successful client
            losses = [(r[0], r[1]) for r in local_evals]
            accuracies = [(r[0], r[2]) for r in local_evals]

            # aggregate local results
            aggregated_loss = FedAvgAgg.aggregate_losses(losses)
            aggregated_accuracy = FedAvgAgg.aggregate_accuracies(accuracies)

            # update model metrics
            self.status.update_agg_model_metrics(r, "eval", aggregated_loss, aggregated_accuracy)
        else:
            self.logger.error("round failed")

    def run_client_fit(self, num_round: int, dev_index: int, config: dict):
        # load training data
        x_train, y_train = self.load_local_data("fit", dev_index)

        # run local data optimizer
        x_train_sub, y_train_sub = self.local_data_optimizer["fit"].optimize(num_round, dev_index,
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
        communication_time = 2 * self.status.con["model"]["tot_weights"] / self.status.con["devs"]["net_speed"][num_round, dev_index]
        energy_consumption = self.config.energy["pow_comp_s"] * computation_time + self.config.energy["pow_net_s"] * communication_time

        # update global configs status
        self.status.update_optimizer_configs(num_round, dev_index, "fit", "local", config)

        # update status
        self.status.update_sim_data(num_round, "fit", dev_index,
                                    computation_time=computation_time,
                                    communication_time=communication_time,
                                    local_iterations=local_iterations,
                                    network_consumption=network_consumption,
                                    energy_consumption=energy_consumption,
                                    accuracy=mean_acc,
                                    loss=mean_loss)

        return x_train_sub.shape[0], model.get_weights(), mean_loss, mean_acc


    def run_client_eval(self, num_round: int, dev_index: int, config: dict):
        # load test data
        x_test, y_test = self.load_local_data("eval", dev_index)

        # run local data optimizer
        x_test_sub, y_test_sub = self.local_data_optimizer["eval"].optimize(num_round, dev_index,
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
        communication_time = self.status.con["model"]["tot_weights"] / self.status.con["devs"]["net_speed"][num_round, dev_index]
        energy_consumption = self.config.energy["pow_comp_s"] * computation_time + self.config.energy["pow_net_s"] * communication_time

        # update global configs status
        self.status.update_optimizer_configs(num_round, dev_index, "eval", "local", config)

        # update status
        self.status.update_sim_data(num_round, "eval", dev_index,
                                    computation_time=computation_time,
                                    communication_time=communication_time,
                                    local_iterations=local_iterations,
                                    network_consumption=network_consumption,
                                    energy_consumption=energy_consumption,
                                    accuracy=accuracy,
                                    loss=loss)

        return x_test_sub.shape[0], loss, accuracy

