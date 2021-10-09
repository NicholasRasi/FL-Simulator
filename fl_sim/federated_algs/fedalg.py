import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from abc import ABC
from fl_sim.configuration import Config
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from fl_sim.status.orchestrator_status import OrchestratorStatus
from fl_sim.utils import FedJob, FedPhase
import statistics as stats


class FedAlg(ABC):

    def __init__(self, status: OrchestratorStatus, config: Config, logger):
        self.status = status
        self.config = config
        self.logger = logger
        self.lock = threading.Lock()
        self.fedjob_queue = multiprocessing.Queue()
        self.fedres_queue = multiprocessing.Queue()
        self.queue_consumer_ps = []
        num_processes = config.simulation["num_processes"]
        for _ in range(num_processes):
            qc_process = multiprocessing.Process(target=self.queue_consumer,
                                                 args=(self.fedjob_queue, self.fedres_queue))
            self.queue_consumer_ps.append(qc_process)
            qc_process.start()

    def get_failed_devs(self, num_round: int):
        return np.where(self.status.con["devs"]["failures"][num_round] == 1)[0]

    def load_local_data(self, fed_phase: FedPhase, dev_index: int):
        x_data = y_data = None
        if fed_phase == FedPhase.FIT:
            x_data = self.x_train[self.status.con["devs"]["local_data"][0][dev_index]]
            y_data = self.y_train[self.status.con["devs"]["local_data"][0][dev_index]]
        elif fed_phase == FedPhase.EVAL:
            x_data = self.x_test[self.status.con["devs"]["local_data"][1][dev_index]]
            y_data = self.y_test[self.status.con["devs"]["local_data"][1][dev_index]]
        return x_data, y_data

    def queue_consumer(self, fedjob_queue, fedres_queue):

        while True:
            fedjob: FedJob = fedjob_queue.get()

            x_data, y_data = fedjob.data
            fedjob.num_examples = x_data.shape[0]

            model_factory = DatasetModelLoaderFactory.get_model_loader(self.config.simulation["model_name"],
                                                                       self.config.devices["num"])
            model = model_factory.get_compiled_model(optimizer=self.config.algorithms["optimizer"],
                                                     metric=self.config.simulation["metric"],
                                                     train_data=fedjob.data)

            loss_func = model_factory.get_loss_function()
            global_weights = fedjob.model_weights
            if fedjob.custom_loss is not None:
                loss_func = fedjob.custom_loss(model_factory.get_loss_function(), model, global_weights)

            model.compile(optimizer=tf.keras.optimizers.get(self.config.algorithms["optimizer"]), run_eagerly=True, metrics=self.config.simulation["metric"], loss=loss_func)

            if fedjob.model_weights is not None:
                model.set_weights(fedjob.model_weights)

            if fedjob.job_type == FedPhase.FIT:
                # fit model
                history = model.fit(x_data, y_data,
                                    epochs=fedjob.config["epochs"],
                                    batch_size=fedjob.config["batch_size"],
                                    verbose=fedjob.config["tf_verbosity"])

                fedjob.mean_metric = stats.mean(history.history[self.config.simulation["metric"]])
                fedjob.mean_loss = stats.mean(history.history['loss'])
                fedjob.model_weights = model.get_weights()

            elif fedjob.job_type == FedPhase.EVAL:
                # evaluate model

                loss, metric = model.evaluate(x_data, y_data, verbose=fedjob.config["tf_verbosity"])
                fedjob.metric = metric
                fedjob.loss = loss

            fedres_queue.put(fedjob)

    def terminate(self):
        for qc_process in self.queue_consumer_ps:
            qc_process.terminate()
