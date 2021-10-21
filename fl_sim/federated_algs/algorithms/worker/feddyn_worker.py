import numpy as np
import tensorflow as tf
import requests
from json_tricks import dumps
import statistics as stats
from tensorflow.keras.callbacks import Callback
from fl_sim.federated_algs.algorithms.worker.fedavg_worker import FedAvgWorker
from fl_sim.federated_algs.loss_functions.feddyn_loss import feddyn_loss
from fl_sim.utils import FedPhase


class FedDynWorker(FedAvgWorker):

    def __init__(self, ip_address, port_number, config, jobs_queue, init_conf):
        super().__init__(ip_address, port_number, config, jobs_queue, init_conf)

        self.current_weights = None
        self.feddyn_gradients = 0

    class GradientsUpdateCallback(Callback):

        def __init__(self, outer_worker, model, prev_gradients, global_weights, alfa_parameter):
            super().__init__()
            self.outer_worker = outer_worker
            self.model = model
            self.global_weights = global_weights
            self.gradients = 0
            self.alfa_parameter = alfa_parameter
            self.prev_gradients = prev_gradients

        def on_epoch_end(self, batch, logs=None):
            weights_diff = 0
            if self.global_weights is not None:
                weights_diff = np.subtract(self.model.get_weights(), self.global_weights)
            self.outer_worker.gradients_update_func(self.prev_gradients - self.alfa_parameter * weights_diff)

    def gradients_update_func(self, updated_gradients):
        self.feddyn_gradients = updated_gradients

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

        global_weights = job["model_weights"]
        self.current_weights = None
        self.feddyn_gradients = 0
        loss_func = feddyn_loss(self.status.model_loader.get_loss_function(), model, job["alfa_parameter"], self.feddyn_gradients, global_weights)

        # compile model
        model.compile(optimizer=tf.keras.optimizers.get(self.status.optimizer), run_eagerly=True, metrics=self.status.metric, loss=loss_func)

        # load weights if not None
        if job["model_weights"] is not None:
            model.set_weights(job["model_weights"])

        # fit model
        gradients_update = self.GradientsUpdateCallback(self, model, self.feddyn_gradients, global_weights, job["alfa_parameter"])
        history = model.fit(x_data, y_data, epochs=job["epochs"], batch_size=job["batch_size"], verbose=job["verbosity"], callbacks=[gradients_update])

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

        global_weights = job["model_weights"]
        self.current_weights = None
        self.feddyn_gradients = 0
        loss_func = feddyn_loss(self.status.model_loader.get_loss_function(), model, job["alfa_parameter"], self.feddyn_gradients, global_weights)

        # compile model
        model.compile(optimizer=tf.keras.optimizers.get(self.status.optimizer), run_eagerly=True, metrics=self.status.metric,
                      loss=loss_func)

        # load weights
        model.set_weights(job["model_weights"])

        # evaluate model
        gradients_update = self.GradientsUpdateCallback(self, model, self.feddyn_gradients, global_weights, job["alfa_parameter"])
        loss, metric = model.evaluate(x_data, y_data, verbose=job["verbosity"], callbacks=gradients_update)

        job_completed = {"metric": metric,
                         "loss": loss,
                         "num_examples": job["num_examples"],
                         "num_round": job["num_round"],
                         "epochs": job["epochs"],
                         "batch_size": job["batch_size"],
                         "dev_index": job["dev_index"]}

        # send results to the orchestrator
        requests.post(self.orchestrator_address + "/send_completed_job", json=dumps(job_completed))







