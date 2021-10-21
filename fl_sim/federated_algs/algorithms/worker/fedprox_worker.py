import tensorflow as tf
import requests
from json_tricks import dumps
import statistics as stats

from tensorflow.python.keras.callbacks import Callback

from fl_sim.federated_algs.algorithms.worker.fedavg_worker import FedAvgWorker
from fl_sim.federated_algs.loss_functions.fed_prox_loss import fed_prox_loss
from fl_sim.utils import FedPhase


class FedProxWorker(FedAvgWorker):

    def __init__(self, ip_address, port_number, config, jobs_queue, init_conf):
        super().__init__(ip_address, port_number, config, jobs_queue, init_conf)
        self.mu_parameter = 1
        self.loss_history = []

    class MuParameterUpdateCallback(Callback):

        def __init__(self, outer_worker):
            super().__init__()
            self.outer_worker = outer_worker

        def on_epoch_end(self, batch, logs=None):
            self.outer_worker.update_mu_parameter(logs["loss"])

    def update_mu_parameter(self, new_loss):

        if len(self.loss_history) > 0 and self.loss_history[len(self.loss_history) - 1] < new_loss:
            self.mu_parameter = min(self.mu_parameter + 0.1, 1)
        else:
            self.mu_parameter = max(self.mu_parameter - 0.02, 0.001)

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
        loss_func = fed_prox_loss(self.status.model_loader.get_loss_function(), model, global_weights, self.mu_parameter)

        # compile model
        model.compile(optimizer=tf.keras.optimizers.get(self.status.optimizer), run_eagerly=True, metrics=self.status.metric, loss=loss_func)

        # load weights if not None
        if job["model_weights"] is not None:
            model.set_weights(job["model_weights"])

        # fit model
        mu_update = self.MuParameterUpdateCallback(self)
        history = model.fit(x_data, y_data, epochs=job["epochs"], batch_size=job["batch_size"], verbose=job["verbosity"], callbacks=[mu_update])

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
        loss_func = fed_prox_loss(self.status.model_loader.get_loss_function(), model, global_weights, self.mu_parameter)

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







