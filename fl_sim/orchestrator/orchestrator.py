import json
import os
import time

from fl_sim.federated_algs.algorithms.federated_algorithm_factory import FederatedAlgorithmFactory
from fl_sim.federated_algs.global_update_optimizer import GlobalUpdateOptimizerFactory
from fl_sim.status.orchestrator_status import OrchestratorStatus
from fl_sim.utils import FedPhase


class Orchestrator:
    def __init__(self, jobs_queue, completed_jobs_queue, workers_queue, config, logger):
        self.jobs_queue = jobs_queue
        self.completed_jobs_queue = completed_jobs_queue
        self.workers_queue = workers_queue
        self.config = config
        self.logger = logger
        self.status = OrchestratorStatus(config=self.config, logger=self.logger)
        print("Initialize orchestrator")

    def start_orchestrator(self):

        run_data = []

        index = 1
        for repetition in range(1, self.config.simulation["repetitions"] + 1):

            time.sleep(2)

            self.logger.info("starting repetition {}/{}".format(repetition, self.config.simulation["repetitions"]))

            self.logger.info("init status...")

            self.logger.info("init federated algorithm")

            self.logger.info("starting training...")
            start_ts = time.time()
            for r in range(self.config.simulation["num_rounds"]):
                self.logger.info("* ROUND %d *" % (r + 1))

                # fit model
                self.logger.info("< FIT >")
                self.model_fit(r)

                # evaluate model
                self.logger.info("< EVAL >")
                self.model_eval(r)

                self.logger.info("eval %s: %.4f | loss: %.4f" %
                                 (self.config.simulation["metric"],
                                  self.status.var["eval"]["model_metrics"]["agg_metric"][r],
                                  self.status.var["eval"]["model_metrics"]["agg_loss"][r]))

                # check if the stopping conditions are met
                if self.status.var["eval"]["model_metrics"]["agg_metric"][r] >= self.config.simulation["stop_conds"][
                    "metric"]:
                    self.logger.info("stopping condition (metric) met. %.4f acc reached" %
                                     self.status.var["eval"]["model_metrics"]["agg_metric"][r])
                    # resize the status
                    self.status.resize_status(r + 1)
                    break
                if self.status.var["eval"]["model_metrics"]["agg_loss"][r] <= self.config.simulation["stop_conds"]["loss"]:
                    self.logger.info("stopping condition (loss) met. %.4f loss reached" %
                                     self.status.var["eval"]["model_metrics"]["agg_loss"][r])
                    self.status.resize_status(r + 1)
                    break
            duration = time.time() - start_ts
            self.logger.info(f"training completed in {duration:.2f} seconds")

            self.logger.info("saving run data")
            run_data.append(self.status.to_dict())

        # export data
        output_dir = self.config.simulation["output_folder"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_dir + "/" + self.config.simulation["output_file"], 'w') as fp:
            json.dump({"status": run_data, "config": self.config.__dict__}, fp)
        self.logger.info("export to {} completed".format(self.config.simulation["output_file"]))

    def model_fit(self, num_round):
        # local fit
        # select devices
        federated_algorithm = FederatedAlgorithmFactory.get_federated_algorithm(self.config.algorithms["federated_algorithm"], self.status, self.config, self.logger)

        dev_indexes = federated_algorithm.select_devs(num_round, FedPhase.FIT)
        # start local fit execution
        created_jobs = 0
        failed_jobs = 0
        failing_devs_indexes = federated_algorithm.get_failed_devs(num_round)
        for dev_index in dev_indexes:
            # run update optimizer
            global_config = federated_algorithm.global_update_optimizer["fit"].optimize(num_round, dev_index, "fit")
            global_config["tf_verbosity"] = self.config.simulation["tf_verbosity"]

            # update global configs status
            self.status.update_optimizer_configs(num_round, dev_index, FedPhase.FIT, "global", global_config["epochs"], global_config["batch_size"], global_config["num_examples"])

            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
                failed_jobs += 1
            else:
                # run client fit
                json_global_weights = None
                if self.status.global_model_weights is not None:
                    json_global_weights = []
                    for x in self.status.global_model_weights:
                        json_global_weights.append(x.tolist())

                global_config = federated_algorithm.global_update_optimizer["fit"].optimize(num_round, dev_index, "fit")
                next_job = {"job_type": "fit",
                            "num_round": int(num_round),
                            "dev_index": int(dev_index),
                            "verbosity": self.config.simulation["tf_verbosity"],
                            "model_weights": json_global_weights,
                            "epochs": global_config["epochs"],
                            "batch_size": global_config["batch_size"],
                            "num_examples": global_config["num_examples"],
                            "custom_loss": None}
                self.jobs_queue.put(json.dumps(next_job))
                created_jobs += 1

        self.logger.info("jobs successful: %d | failed: %d" % (created_jobs, failed_jobs))

        # wait until all the results are available
        while self.completed_jobs_queue.qsize() < created_jobs:
            time.sleep(1)
        print("Ho ricevuto ", self.completed_jobs_queue.qsize(), "fit jobs completati.")

        local_fits = federated_algorithm.get_fit_results(self.completed_jobs_queue, created_jobs)

        if len(local_fits) > 0:  # at least one successful client
            weights = [(r[0], r[1]) for r in local_fits]
            losses = [(r[0], r[2]) for r in local_fits]
            accuracies = [(r[0], r[3]) for r in local_fits]

            # aggregate local results
            aggregated_weights = federated_algorithm.aggregator["fit"].aggregate_fit(weights)
            aggregated_loss = federated_algorithm.aggregator["fit"].aggregate_losses(losses)
            aggregated_metrics = federated_algorithm.aggregator["fit"].aggregate_accuracies(accuracies)

            # update global model and model metrics
            self.status.global_model_weights = aggregated_weights
            self.status.update_agg_model_metrics(num_round, FedPhase.FIT, aggregated_loss, aggregated_metrics)
        else:
            self.logger.error("round failed")

    def model_eval(self, num_round):
        # local evaluation
        # select devices
        federated_algorithm = FederatedAlgorithmFactory.get_federated_algorithm(self.config.algorithms["federated_algorithm"], self.status, self.config, self.logger)

        dev_indexes = federated_algorithm.select_devs(num_round, FedPhase.EVAL)

        # start local eval execution
        created_jobs = 0
        failed_jobs = 0
        failing_devs_indexes = federated_algorithm.get_failed_devs(num_round)
        for dev_index in dev_indexes:
            # run global update optimizer
            global_config = federated_algorithm.global_update_optimizer["eval"].optimize(num_round, dev_index, "eval")
            global_config["tf_verbosity"] = self.config.simulation["tf_verbosity"]
            # update global configs status
            self.status.update_optimizer_configs(num_round, dev_index, FedPhase.EVAL, "global", global_config["epochs"], global_config["batch_size"], global_config["num_examples"])
            # check if device fails
            if dev_index in failing_devs_indexes:
                pass
                # self.logger.error("dev fails: {}".format(dev_index))
                failed_jobs += 1
            else:
                # run client eval
                # run client fit
                global_weights = self.status.global_model_weights
                global_config = federated_algorithm.global_update_optimizer["eval"].optimize(num_round, dev_index, "eval")

                json_global_weights = []
                for x in global_weights:
                    json_global_weights.append(x.tolist())

                next_job = {"job_type": "eval",
                            "num_round": int(num_round),
                            "dev_index": int(dev_index),
                            "verbosity": self.config.simulation["tf_verbosity"],
                            "model_weights": json_global_weights,
                            "epochs": global_config["epochs"],
                            "batch_size": global_config["batch_size"],
                            "num_examples": global_config["num_examples"],
                            "custom_loss": None}

                self.jobs_queue.put(json.dumps(next_job))
                created_jobs += 1

        self.logger.info("jobs successful: %d | failed: %d" % (created_jobs, failed_jobs))

        # wait until all the results are available
        while self.completed_jobs_queue.qsize() < created_jobs:
            time.sleep(1)
        print("Ho ricevuto ", self.completed_jobs_queue.qsize(), "eval jobs completati.")
        local_evals = federated_algorithm.get_eval_results(self.completed_jobs_queue, created_jobs)

        if len(local_evals) > 0:  # at least one successful client
            losses = [(r[0], r[1]) for r in local_evals]
            accuracies = [(r[0], r[2]) for r in local_evals]

            # aggregate local results
            aggregated_loss = federated_algorithm.aggregator["eval"].aggregate_losses(losses)
            aggregated_accuracy = federated_algorithm.aggregator["eval"].aggregate_accuracies(accuracies)
            # update model metrics
            self.status.update_agg_model_metrics(num_round, FedPhase.EVAL, aggregated_loss, aggregated_accuracy)
        else:
            self.logger.error("round failed")

