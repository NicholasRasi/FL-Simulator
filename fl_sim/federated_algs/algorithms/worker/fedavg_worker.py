import numpy as np
import tensorflow as tf
import requests
from json_tricks import dumps, loads
import statistics as stats
from fl_sim.status.worker_status import WorkerStatus
from fl_sim.utils import FedPhase


class FedAvgWorker:
    def __init__(self, ip_address, port_number, config, jobs_queue, init_conf):
        self.ip_address = ip_address
        self.port_number = port_number
        self.config = config
        self.jobs_queue = jobs_queue
        self.orchestrator_address = "http://127.0.0.1:8000"
        self.status = WorkerStatus(config)
        self.status.initialize_global_fields(init_conf)

    def load_local_data(self, fed_phase: FedPhase, dev_index: int):
        x_data = y_data = None
        if fed_phase == FedPhase.FIT:
            x_data = self.status.x_train[self.status.train_indexes[dev_index]]
            y_data = self.status.y_train[self.status.train_indexes[dev_index]]
        elif fed_phase == FedPhase.EVAL:
            x_data = self.status.x_test[self.status.eval_indexes[dev_index]]
            y_data = self.status.y_test[self.status.eval_indexes[dev_index]]
        return x_data, y_data

    def handle_fit_job(self, job):
        # load data
        x_train, y_train = self.load_local_data(FedPhase.FIT, job["dev_index"])

        # run local data optimizer
        x_data, y_data = self.status.local_optimizer_fit.optimize(job["num_round"], job["dev_index"],
                                                                             job["num_examples"],
                                                                             self.status.dataset,
                                                                             self.status.dev_num,
                                                                             (x_train, y_train), FedPhase.FIT)

        # load model
        model = self.status.model_loader.get_compiled_model(optimizer=self.status.optimizer, metric=self.status.metric, train_data=(x_data, y_data))
        loss_func = self.status.model_loader.get_loss_function()

        # compile model
        model.compile(optimizer=tf.keras.optimizers.get(self.status.optimizer), run_eagerly=True, metrics=self.status.metric, loss=loss_func)

        # load weights if not None
        if job["model_weights"] is not None:
            model.set_weights(job["model_weights"])

        # fit model
        history = model.fit(x_data, y_data, epochs=job["epochs"], batch_size=job["batch_size"], verbose=job["verbosity"])

        mean_metric = stats.mean(history.history[self.status.metric])
        mean_loss = stats.mean(history.history['loss'])
        model_weights = model.get_weights()

        job_completed = {"mean_metric": mean_metric,
                         "mean_loss": mean_loss,
                         "model_weights": model_weights,
                         "num_examples": job["num_examples"],
                         "num_round": job["num_round"],
                         "dev_index": job["dev_index"],
                         "epochs": job["epochs"],
                         "batch_size": job["batch_size"]}

        # send results to the orchestrator
        requests.post(self.orchestrator_address + "/send_completed_job", json=dumps(job_completed, conv_str_byte=True))

    def handle_eval_job(self, job):
        # load data
        x_train, y_train = self.load_local_data(FedPhase.EVAL, job["dev_index"])

        # run local data optimizer
        x_data, y_data = self.status.local_optimizer_fit.optimize(job["num_round"], job["dev_index"],
                                                           job["num_examples"],
                                                           self.status.dataset,
                                                           self.status.dev_num,
                                                           (x_train, y_train), FedPhase.EVAL)

        # load model
        model = self.status.model_loader.get_compiled_model(optimizer=self.status.optimizer, metric=self.status.metric, train_data=(x_data, y_data))

        loss_func = self.status.model_loader.get_loss_function()

        # compile model
        model.compile(optimizer=tf.keras.optimizers.get(self.status.optimizer), run_eagerly=True, metrics=self.status.metric,
                      loss=loss_func)

        # load weights
        model.set_weights(job["model_weights"])

        # evaluate model
        loss, metric = model.evaluate(x_data, y_data, verbose=job["verbosity"])

        job_completed = {"metric": metric,
                         "loss": loss,
                         "num_examples": job["num_examples"],
                         "num_round": job["num_round"],
                         "epochs": job["epochs"],
                         "batch_size": job["batch_size"],
                         "dev_index": job["dev_index"]}

        # send results to the orchestrator
        requests.post(self.orchestrator_address + "/send_completed_job", json=dumps(job_completed))







