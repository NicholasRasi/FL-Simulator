import time
from fl_sim.status import Status
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from json_tricks import dump
import os
from fl_sim.federated_algs.algorithms.federated_algorithm_factory import FederatedAlgorithmFactory
from fl_sim.federated_algs.algorithms.fedavg import FedAvg


class Simulator:

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

        self.run_data = []
        self.output_dir = self.config.simulation["output_folder"]
        self.data = DatasetModelLoaderFactory.get_model_loader(config.simulation["model_name"], config.devices["num"])\
            .get_dataset(config.data["mislabelling_percentage"])

        self.fed_alg = None

    def save_run_data(self, status):
        self.run_data.append(status.to_dict())

    def export_all_data(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.output_dir + "/" + self.config.simulation["output_file"], 'w') as fp:
            dump({"status": self.run_data, "config": self.config.__dict__}, fp)

    def run(self):
        for repetition in range(1, self.config.simulation["repetitions"] + 1):
            self.logger.info("starting repetition {}/{}".format(repetition, self.config.simulation["repetitions"]))

            self.logger.info("init status...")
            status = Status(config=self.config, logger=self.logger)

            self.logger.info("init federated algorithm")
            self.fed_alg = FederatedAlgorithmFactory.get_federated_algorithm(self.config.algorithms["federated_algorithm"] ,status, self.data, self.config, self.logger)
            #self.fed_alg = FedAvg(status, self.data, self.config, self.logger)

            self.logger.info("starting training...")
            start_ts = time.time()
            for r in range(self.config.simulation["num_rounds"]):
                self.logger.info("* ROUND %d *" % (r+1))

                # fit model
                self.logger.info("< FIT >")
                self.fed_alg.model_fit(r)

                # evaluate model
                self.logger.info("< EVAL >")
                self.fed_alg.model_eval(r)

                self.logger.info("eval %s: %.4f | loss: %.4f" %
                                 (self.config.simulation["metric"],
                                  status.var["eval"]["model_metrics"]["agg_metric"][r],
                                  status.var["eval"]["model_metrics"]["agg_loss"][r]))

                # check if the stopping conditions are met
                if status.var["eval"]["model_metrics"]["agg_metric"][r] >= self.config.simulation["stop_conds"]["metric"]:
                    self.logger.info("stopping condition (metric) met. %.4f acc reached" % status.var["eval"]["model_metrics"]["agg_metric"][r])
                    # resize the status
                    status.resize_status(r+1)
                    break
                if status.var["eval"]["model_metrics"]["agg_loss"][r] <= self.config.simulation["stop_conds"]["loss"]:
                    self.logger.info("stopping condition (loss) met. %.4f loss reached" % status.var["eval"]["model_metrics"]["agg_loss"][r])
                    status.resize_status(r+1)
                    break
            self.fed_alg.terminate()
            duration = time.time() - start_ts
            self.logger.info(f"training completed in {duration:.2f} seconds")

            self.logger.info("saving run data")
            self.save_run_data(status)

        self.export_all_data()
        self.logger.info("export to {} completed".format(self.config.simulation["output_file"]))
