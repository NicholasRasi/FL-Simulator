from json_tricks import dump
import os
import time
from fl_sim.federated_algs.algorithms.orchestrator.orchestrator_algorithm_factory import OrchestratorAlgorithmFactory


class Orchestrator:
    def __init__(self, jobs_queue, completed_jobs_queue, lock, workers_queue, config, logger, status):
        self.jobs_queue = jobs_queue
        self.completed_jobs_queue = completed_jobs_queue
        self.workers_queue = workers_queue
        self.config = config
        self.lock = lock
        self.logger = logger
        self.status = status
        self.federated_algorithm = OrchestratorAlgorithmFactory.get_federated_algorithm(self.config.algorithms["federated_algorithm"], self.status, self.config, self.logger, self.jobs_queue, self.completed_jobs_queue, self.workers_queue, self.lock)

    def start_orchestrator(self):
        run_data = []

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
                if self.status.var["eval"]["model_metrics"]["agg_metric"][r] >= self.config.simulation["stop_conds"]["metric"]:
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
            self.status.reinitialize_status()

        self.export_data(run_data)

    def export_data(self, run_data):
        # export data
        output_dir = "../../" + self.config.simulation["output_folder"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_dir + "/" + self.config.simulation["output_file"], 'w') as fp:
            dump({"status": run_data, "config": self.config.__dict__}, fp)
        self.logger.info("export to " + self.config.simulation["output_file"] + " completed.")

    def model_fit(self, num_round):
        # select federated algorithm
        self.federated_algorithm.model_fit(num_round)

    def model_eval(self, num_round):
        # select federated algorithm
        self.federated_algorithm.model_eval(num_round)

