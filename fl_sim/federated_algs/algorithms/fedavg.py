import json
import time

import numpy

from fl_sim.configuration import Config
from fl_sim.federated_algs.fedalg import FedAlg
from fl_sim.federated_algs.clients_selector import ClientsSelectorFactory
from fl_sim.federated_algs.aggregation_strategy.aggregation_strategy_factory import AggregationStrategyFactory
from fl_sim.federated_algs.global_update_optimizer import GlobalUpdateOptimizerFactory
from fl_sim.status.orchestrator_status import OrchestratorStatus
from fl_sim.utils import FedJob, FedPhase


class FedAvg(FedAlg):

    def __init__(self, status: OrchestratorStatus, config: Config, logger):
        super().__init__(status, config, logger)

        self.clients_selector = None
        self.aggregator = None
        self.global_update_optimizer = None
        self.local_data_optimizer = None

        self.init_optimizers()

    def init_optimizers(self):
        self.clients_selector = {
            "fit": ClientsSelectorFactory.get_clients_selector(self.config.algorithms["fit"]["selection"], self.config,
                                                               self.status, self.logger),
            "eval": ClientsSelectorFactory.get_clients_selector(self.config.algorithms["eval"]["selection"],
                                                                self.config,
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
        #self.local_data_optimizer = {
        #    "fit": LocalDataOptimizerFactory.get_optimizer(self.config.algorithms["fit"]["data"], self.status,
        #                                                   self.logger),
        #    "eval": LocalDataOptimizerFactory.get_optimizer(self.config.algorithms["eval"]["data"], self.status,
        #                                                    self.logger),
        #}
        self.aggregator = {
            "fit": AggregationStrategyFactory.get_aggregation_strategy(self.config.algorithms["fit"]["aggregation"], self.status, self.config.algorithms["fit"]["data"], self.config, self.logger),
            "eval": AggregationStrategyFactory.get_aggregation_strategy(self.config.algorithms["eval"]["aggregation"], self.status, self.config.algorithms["eval"]["data"], self.config, self.logger)
        }

    def select_devs(self, num_round: int, fed_phase: FedPhase):
        phase = fed_phase.value
        # get dev indexes
        dev_indexes = self.clients_selector[phase].select_devices(num_round)
        # update status
        self.status.var[phase]["devs"]["selected"][num_round, dev_indexes] = 1
        return dev_indexes

    def model_eval(self, num_round: int):
        # local evaluation
        # select devices
        dev_indexes = self.select_devs(num_round, FedPhase.EVAL)

        # start local eval execution
        created_jobs = 0
        failed_jobs = 0
        failing_devs_indexes = self.get_failed_devs(num_round)
        for dev_index in dev_indexes:
            # run global update optimizer
            global_config = self.global_update_optimizer["eval"].optimize(num_round, dev_index, "eval")
            global_config["tf_verbosity"] = self.config.simulation["tf_verbosity"]

            # update global configs status
            self.status.update_optimizer_configs(num_round, dev_index, FedPhase.EVAL, "global", global_config)

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
                failed_jobs += 1
            else:
                # run client eval
                self.put_client_job_eval(num_round, dev_index, global_config)
                created_jobs += 1

        self.logger.info("jobs successful: %d | failed: %d" % (created_jobs, failed_jobs))

        # wait until all the results are available
        while self.fedres_queue.qsize() < created_jobs:
            time.sleep(1)
        local_evals = self.get_eval_results(created_jobs)

        if len(local_evals) > 0:  # at least one successful client
            losses = [(r[0], r[1]) for r in local_evals]
            accuracies = [(r[0], r[2]) for r in local_evals]

            # aggregate local results
            aggregated_loss = self.aggregator["eval"].aggregate_losses(losses)
            aggregated_accuracy = self.aggregator["eval"].aggregate_accuracies(accuracies)

            # update model metrics
            self.status.update_agg_model_metrics(num_round, FedPhase.EVAL, aggregated_loss, aggregated_accuracy)
        else:
            self.logger.error("round failed")

    def put_client_job_fit(self, num_round: int, dev_index: int, job_config: dict):
        # load training data
        x_train, y_train = self.load_local_data(FedPhase.FIT, dev_index)

        # run local data optimizer
        x_train_sub, y_train_sub = self.local_data_optimizer["fit"].optimize(num_round, dev_index,
                                                                             job_config["num_examples"],
                                                                             self.config.simulation["model_name"],
                                                                             self.config.devices["num"],
                                                                             (x_train, y_train))
        job_config["num_examples"] = x_train_sub.shape[0]

        # update local model to global model
        # model = self.status.con["devs"]["local_models"][dev_index]
        # model.set_weights(self.status.global_model.get_weights())

        # get global model weights (initially None)
        model_weights = self.status.global_model_weights

        # create fit FedJob
        fedjob = FedJob(job_type=FedPhase.FIT, num_round=num_round, dev_index=dev_index,
                        data=(x_train_sub, y_train_sub), config=job_config, model_weights=model_weights, custom_loss=None)
        self.fedjob_queue.put(fedjob)

    def put_client_job_eval(self, num_round: int, dev_index: int, job_config: dict):
        # load test data
        x_test, y_test = self.load_local_data(FedPhase.EVAL, dev_index)

        # run local data optimizer
        x_test_sub, y_test_sub = self.local_data_optimizer["eval"].optimize(num_round, dev_index,
                                                                            job_config["num_examples"],
                                                                            self.config.simulation["model_name"],
                                                                            self.config.devices["num"],
                                                                            (x_test, y_test))
        job_config["num_examples"] = x_test_sub.shape[0]

        # get global model weights
        model_weights = self.status.global_model_weights

        # evaluate model
        fedjob = FedJob(job_type=FedPhase.EVAL, num_round=num_round, dev_index=dev_index,
                        data=(x_test_sub, y_test_sub), config=job_config, model_weights=model_weights, custom_loss=None)
        self.fedjob_queue.put(fedjob)

    def get_fit_results(self, completed_jobs_queue, created_jobs):
        fit_results = []
        for _ in range(created_jobs):
            fedres = completed_jobs_queue.get()

            array_model_weights = []
            for x in fedres["model_weights"]:
                array_model_weights.append(numpy.array(x))

            if self.status.con["model"]["tot_weights"] is None:
                self.status.con["model"]["tot_weights"] = sum([len(w_list) for w_list in array_model_weights])
            # update model weights with the new computed ones
            self.status.con["devs"]["local_models_weights"][fedres.get("dev_index")] = array_model_weights

            # compute metrics
            local_iterations = fedres.get("epochs") * fedres.get("num_examples") / fedres.get("batch_size")
            computation_time = local_iterations / self.status.con["devs"]["ips"][fedres.get("dev_index")]
            network_consumption = 2 * self.status.con["model"]["tot_weights"]
            communication_time = network_consumption /\
                                 self.status.con["devs"]["net_speed"][fedres.get("num_round"), fedres.get("dev_index")]
            energy_consumption = self.config.energy["pow_comp_s"] * computation_time +\
                                 self.config.energy["pow_net_s"] * communication_time

            # update global configs status
            self.status.update_optimizer_configs(fedres.get("num_round"), fedres.get("dev_index"), FedPhase.FIT,
                                                 "local", fedres.get("epochs"), fedres.get("batch_size"), fedres.get("num_examples"))

            # update status
            self.status.update_sim_data(fedres.get("num_round"), FedPhase.FIT, fedres.get("dev_index"),
                                        computation_time=computation_time,
                                        communication_time=communication_time,
                                        local_iterations=local_iterations,
                                        network_consumption=network_consumption,
                                        energy_consumption=energy_consumption,
                                        metric=fedres.get("mean_metric"),
                                        loss=fedres.get("mean_loss"))

            fit_results.append((fedres.get("num_examples"), array_model_weights, fedres.get("mean_loss"), fedres.get("mean_metric")))
        return fit_results

    def get_eval_results(self, completed_jobs_queue, created_jobs):
        eval_results = []
        for _ in range(created_jobs):
            fedres = completed_jobs_queue.get()

            # compute metrics
            local_iterations = fedres["num_examples"] / fedres["batch_size"]
            computation_time = local_iterations / self.status.con["devs"]["ips"][fedres["dev_index"]]
            network_consumption = self.status.con["model"]["tot_weights"]
            communication_time = network_consumption /\
                                 self.status.con["devs"]["net_speed"][fedres["num_round"], fedres["dev_index"]]
            energy_consumption = self.config.energy["pow_comp_s"] * computation_time + self.config.energy[
                "pow_net_s"] * communication_time

            # update global configs status
            self.status.update_optimizer_configs(fedres["num_round"], fedres["dev_index"], FedPhase.EVAL, "local", fedres.get("epochs"), fedres.get("batch_size"), fedres.get("num_examples"))

            # update status
            self.status.update_sim_data(fedres["num_round"], FedPhase.EVAL, fedres["dev_index"],
                                        computation_time=computation_time,
                                        communication_time=communication_time,
                                        local_iterations=local_iterations,
                                        network_consumption=network_consumption,
                                        energy_consumption=energy_consumption,
                                        metric=fedres["metric"],
                                        loss=fedres["loss"])
            eval_results.append((fedres["num_examples"], fedres["loss"], fedres["metric"]))
        return eval_results
