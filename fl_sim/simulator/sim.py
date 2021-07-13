import time
from fl_sim.status import Status
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader import DatasetModelLoader
from json_tricks import dump
from tqdm import trange
import os
from fl_sim.federated_algs.fedavg import FedAvg


class Simulator:

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

        self.run_data = []
        self.output_dir = self.config.simulation["output_folder"]
        self.data = DatasetModelLoader(config.simulation["model_name"]).get_dataset()

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
            self.fed_alg = FedAvg(status, self.data, self.config, self.logger)

            self.logger.info("starting training...")
            start_ts = time.time()
            for r in range(self.config.simulation["num_rounds"]):
                self.logger.info("* running round %d *" % (r+1))

                # fit model
                self.fed_alg.model_fit(r)

                # evaluate model
                self.fed_alg.model_eval(r)

                self.logger.info("eval accuracy: %.4f | loss: %.4f" %
                                 (status.var["eval"]["model_metrics"]["agg_accuracy"][r],
                                  status.var["eval"]["model_metrics"]["agg_loss"][r]))

                # check if the stopping conditions are met
                if status.var["eval"]["model_metrics"]["agg_accuracy"][r] >= self.config.simulation["stop_conds"]["accuracy"]:
                    self.logger.info("stopping condition (accuracy) met. %.4f acc reached" % status.var["eval"]["model_metrics"]["agg_accuracy"][r])
                    break
                if status.var["eval"]["model_metrics"]["agg_loss"][r] <= self.config.simulation["stop_conds"]["loss"]:
                    self.logger.info("stopping condition (loss) met. %.4f loss reached" % status.var["eval"]["model_metrics"]["agg_loss"][r])
                    break
            self.fed_alg.terminate()
            duration = time.time() - start_ts
            self.logger.info(f"training completed in {duration:.2f} seconds")

            self.logger.info("saving run data")
            self.save_run_data(status)

        self.export_all_data()
        self.logger.info("export to {} completed".format(self.config.simulation["output_file"]))
