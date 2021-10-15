import logging
import tensorflow as tf
from time import sleep
import requests
from json_tricks import dumps, loads
import statistics as stats
from fl_sim.federated_algs.loss_functions.custom_loss_factory import CustomLossFactory
from fl_sim.status.worker_status import WorkerStatus
from fl_sim.utils import FedPhase


class Worker:
    def __init__(self, ip_address, port_number, config, jobs_queue):
        self.ip_address = ip_address
        self.port_number = port_number
        self.config = config
        self.jobs_queue = jobs_queue
        self.orchestrator_address = "http://127.0.0.1:8000"
        self.status = WorkerStatus(config)

    def start_worker(self, orchestrator_empty_queue):
        # register worker
        response = requests.post(self.orchestrator_address + "/register_worker", json={"ip_address": self.ip_address, 'port_number': self.port_number})
        self.status.initialize_global_fields(loads(response.text))

        if response.status_code == 200:
            logging.info("Registration was successful.")
            init_conf = response.json()
            del init_conf["train_indexes"]
            del init_conf["eval_indexes"]
            logging.info("Init configuration: " + str(init_conf))

            # handle available jobs
            while True:

                if orchestrator_empty_queue.value is False:
                    next_job = requests.get(self.orchestrator_address+"/get_next_jobs")

                    if len(next_job.text) > 0:

                        json_next_job = loads(next_job.text)
                        logging.info(str("Handle " + json_next_job["job_type"] + " job , num round " + str(json_next_job["num_round"]) + " dev index " + str(json_next_job["dev_index"])))
                        if json_next_job["job_type"] == 'fit':
                            self.handle_fit_job(json_next_job)
                        else:
                            self.handle_eval_job(json_next_job)
                    else:
                        sleep(1)
                else:
                    sleep(1)
        else:
            logging.info("Registration refused.")

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

        global_weights = job["model_weights"]
        if job["custom_loss"] is not None:
            loss_func = CustomLossFactory.get_custom_loss(job["custom_loss"])(self.status.model_loader.get_loss_function(), model, global_weights)

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

        global_weights = job["model_weights"]
        if job["custom_loss"] is not None:
            loss_func = CustomLossFactory.get_custom_loss(job["custom_loss"])(self.status.model_loader.get_loss_function(), model,
                                                                              global_weights)

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







