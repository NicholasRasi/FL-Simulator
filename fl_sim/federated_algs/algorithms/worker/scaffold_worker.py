import numpy as np
import requests
from json_tricks import dumps
import statistics as stats
from tensorflow import keras
from fl_sim.federated_algs.algorithms.worker.fedavg_worker import FedAvgWorker
from fl_sim.federated_algs.training_optimizer.scaffold_optimizer import SCAFFOLD_optimizer
from fl_sim.utils import FedPhase


class SCAFFOLDWorker(FedAvgWorker):

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
        optimizer = SCAFFOLD_optimizer(global_control_variate=job["global_control_variate"], local_control_variate=job["local_control_variate"], num_layers=len(model.get_weights()))

        # compile model
        model.compile(optimizer=optimizer, run_eagerly=True, metrics=self.status.metric, loss=loss_func)

        # load weights if not None
        if job["model_weights"] is not None:
            model.set_weights(job["model_weights"])

        # fit model
        history = model.fit(x_data, y_data, epochs=job["epochs"], batch_size=job["batch_size"], verbose=job["verbosity"])

        mean_metric = stats.mean(history.history[self.status.metric])
        mean_loss = stats.mean(history.history['loss'])
        model_weights = model.get_weights()

        local_iter = job["batch_size"] / (job["epochs"] * job["num_examples"] * keras.backend.eval(model.optimizer.lr))
        if job["model_weights"] is not None:
            weights_delta = np.subtract(job["model_weights"], model_weights)
            delta_variate = - job["global_control_variate"] + local_iter * weights_delta
        else:
            delta_variate = 0

        job_completed = {"mean_metric": mean_metric,
                         "mean_loss": mean_loss,
                         "model_weights": model_weights,
                         "num_examples": job["num_examples"],
                         "num_round": job["num_round"],
                         "dev_index": job["dev_index"],
                         "epochs": job["epochs"],
                         "batch_size": job["batch_size"],
                         "local_control_delta": delta_variate}

        # send results to the orchestrator
        requests.post(self.orchestrator_address + "/send_completed_job", json=dumps(job_completed, conv_str_byte=True))
