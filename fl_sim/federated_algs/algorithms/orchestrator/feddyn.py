import time
import numpy as np
import requests
from json_tricks import loads, dumps
from fl_sim import FedAvg
from fl_sim.configuration import Config
from fl_sim.federated_algs.aggregation_strategy.feddyn_agg import FedDynAgg
from fl_sim.status.orchestrator_status import OrchestratorStatus
from fl_sim.utils import FedPhase


class FedDyn(FedAvg):

    def __init__(self, status: OrchestratorStatus, config: Config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock):
        super().__init__(status, config, logger,  jobs_queue, completed_jobs_queue, workers_queue, lock)

        self.aggregator = FedDynAgg(status, config, logger)
        self.alfa_parameter = 0.01
        self.h = 0

    def update_h(self, results):
        term = 0
        if self.status.global_model_weights is not None:
            for result in results:
                diff = np.subtract(result[1], self.status.global_model_weights)
                term += diff
            term = term * self.alfa_parameter / len(results)
            self.h = self.h - term

    def model_fit(self, num_round):
        # select devices
        dev_indexes = self.select_devs(num_round, FedPhase.FIT)

        created_jobs = 0
        failed_jobs = 0
        failing_devs_indexes = self.get_failed_devs(num_round)
        for dev_index in dev_indexes:
            # run update optimizer
            global_config = self.global_update_optimizer["fit"].optimize(num_round, dev_index, "fit")
            global_config["tf_verbosity"] = self.config.simulation["tf_verbosity"]

            # update global configs status
            self.status.update_optimizer_configs(num_round, dev_index, FedPhase.FIT, "global", global_config["epochs"], global_config["batch_size"], global_config["num_examples"])

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                failed_jobs += 1
            else:
                global_config = self.global_update_optimizer["fit"].optimize(num_round, dev_index, "fit")
                self.put_client_job_fit(num_round, dev_index, global_config)
                created_jobs += 1

        self.logger.info("jobs successful: %d | failed: %d" % (created_jobs, failed_jobs))

        # wait until all the results are available
        while self.completed_jobs_queue.qsize() < created_jobs:
            time.sleep(1)
        local_fits = self.get_fit_results(self.completed_jobs_queue, created_jobs)

        self.update_h(local_fits)

        if len(local_fits) > 0:  # at least one successful client
            weights = [(r[1]) for r in local_fits]
            losses = [(r[0], r[2]) for r in local_fits]
            accuracies = [(r[0], r[3]) for r in local_fits]

            # aggregate local results
            aggregated_weights = self.aggregator.aggregate_fit(weights)
            aggregated_loss = self.aggregator.aggregate_losses(losses)
            aggregated_metrics = self.aggregator.aggregate_accuracies(accuracies)

            # update global model and model metrics
            self.status.global_model_weights = aggregated_weights
            self.status.update_agg_model_metrics(num_round, FedPhase.FIT, aggregated_loss, aggregated_metrics)
        else:
            self.logger.error("round failed")

    def put_client_job_fit(self, num_round: int, dev_index: int, global_config: dict):

        next_job = {"job_type": "fit",
                    "num_round": int(num_round),
                    "dev_index": int(dev_index),
                    "verbosity": self.config.simulation["tf_verbosity"],
                    "model_weights": self.status.global_model_weights,
                    "epochs": global_config["epochs"],
                    "batch_size": global_config["batch_size"],
                    "alfa_parameter": self.alfa_parameter,
                    "num_examples": global_config["num_examples"]}

        self.lock.acquire()
        if self.jobs_queue.qsize() == 0:
            for worker in self.workers_queue:
                requests.post("http://" + worker[0] + ":" + str(worker[1]) + "/notify_available_jobs")
        self.jobs_queue.put(dumps(next_job))
        self.lock.release()

    def put_client_job_eval(self, num_round: int, dev_index: int, global_config: dict):

        next_job = {"job_type": "eval",
                    "num_round": int(num_round),
                    "dev_index": int(dev_index),
                    "verbosity": self.config.simulation["tf_verbosity"],
                    "model_weights": self.status.global_model_weights,
                    "epochs": global_config["epochs"],
                    "batch_size": global_config["batch_size"],
                    "alfa_parameter": self.alfa_parameter,
                    "num_examples": global_config["num_examples"]}

        self.lock.acquire()
        if self.jobs_queue.qsize() == 0:
            for worker in self.workers_queue:
                requests.post("http://" + worker[0] + ":" + str(worker[1]) + "/notify_available_jobs")
        self.jobs_queue.put(dumps(next_job))
        self.lock.release()

    def get_fit_results(self, completed_jobs_queue, created_jobs):
        fit_results = []
        for _ in range(created_jobs):
            fedres = loads(completed_jobs_queue.get())

            if self.status.con["model"]["tot_weights"] is None:
                self.status.con["model"]["tot_weights"] = sum([w_list.size for w_list in fedres.get("model_weights")])

            # update model weights with the new computed ones
            self.status.con["devs"]["local_models_weights"][fedres.get("dev_index")] = fedres.get("model_weights")

            # compute metrics
            local_iterations = fedres.get("epochs") * fedres.get("num_examples") / fedres.get("batch_size")
            computation_time = local_iterations / self.status.con["devs"]["ips"][fedres.get("dev_index")]
            network_consumption = 2 * self.status.con["model"]["tot_weights"]
            communication_time = network_consumption / \
                                 self.status.con["devs"]["net_speed"][fedres.get("num_round"), fedres.get("dev_index")]
            energy_consumption = self.config.energy["pow_comp_s"] * computation_time + \
                                 self.config.energy["pow_net_s"] * communication_time

            # update global configs status
            self.status.update_optimizer_configs(fedres.get("num_round"), fedres.get("dev_index"), FedPhase.FIT,
                                                 "local", fedres.get("epochs"), fedres.get("batch_size"),
                                                 fedres.get("num_examples"))

            # update status
            self.status.update_sim_data(fedres.get("num_round"), FedPhase.FIT, fedres.get("dev_index"),
                                        computation_time=computation_time,
                                        communication_time=communication_time,
                                        local_iterations=local_iterations,
                                        network_consumption=network_consumption,
                                        energy_consumption=energy_consumption,
                                        metric=fedres.get("mean_metric"),
                                        loss=fedres.get("mean_loss"))

            fit_results.append(
                (fedres.get("num_examples"), fedres.get("model_weights"), fedres.get("mean_loss"),
                 fedres.get("mean_metric")))
        return fit_results
