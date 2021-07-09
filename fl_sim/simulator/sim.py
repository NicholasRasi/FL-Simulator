import time
from fl_sim import Status
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader import DatasetModelLoader
from fl_sim.federated_algs import FedAvg
from json_tricks import dump
from tqdm import trange
import os


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

            self.logger.info("init fed alg")
            self.fed_alg = FedAvg(status, self.data, self.config, self.logger)

            self.logger.info("starting training...")
            start_ts = time.time()
            for r in trange(self.config.simulation["num_rounds"]):
                self.fed_alg.model_fit(r)
                self.fed_alg.model_eval(r)
            duration = time.time() - start_ts
            self.logger.info(f"training completed in {duration} seconds")

            self.logger.info("saving run data")
            self.save_run_data(status)

        self.export_all_data()
        self.logger.info("export to {} completed".format(self.config.simulation["output_file"]))
