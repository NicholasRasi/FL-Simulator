import numpy
import numpy as np
import tensorflow as tf
from time import sleep
import requests
import statistics as stats
from fl_sim import Status
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from fl_sim.federated_algs.local_data_optimizer import LocalDataOptimizerFactory
from fl_sim.utils import FedPhase


class Worker:
    def __init__(self, config, jobs_queue):
        self.config = config
        self.jobs_queue = jobs_queue
        self.orchestrator_address = "http://127.0.0.1:8000"
        self.dataset = None
        self.dev_num = None
        self.fedalg = None
        self.verbosity = None
        self.optimizer = None
        self.metric = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.non_iid_partitions = None
        self.model_loader = None
        self.data_sizes = None
        self.train_indexes = None
        self.eval_indexes = None
        self.local_optimizer_fit = None
        self.local_optimizer_eval = None
        np.random.seed(self.config.simulation["seed"])

    def fillup_registration_fields(self, json_fields):
        self.dataset = json_fields["dataset"]
        self.dev_num = json_fields["dev_num"]
        self.fedalg = json_fields["fedalg"]
        self.verbosity = json_fields["verbosity"]
        self.optimizer = json_fields["optimizer"]
        self.metric = json_fields["metric"]
        self.non_iid_partitions = json_fields["non_iid_partitions"]
        self.model_loader = DatasetModelLoaderFactory.get_model_loader(self.dataset, self.dev_num)
        self.local_optimizer_fit = LocalDataOptimizerFactory.get_optimizer(json_fields["local_optimizer_fit"])
        self.local_optimizer_eval = LocalDataOptimizerFactory.get_optimizer(json_fields["local_optimizer_eval"])
        self.data_sizes = Status.randint(mean=self.config.data["num_examples_mean"],
                                       var=self.config.data["num_examples_var"],
                                       size=self.dev_num, dtype=int)
        self.x_train, self.y_train, self.x_test, self.y_test = DatasetModelLoaderFactory.get_model_loader(self.dataset, self.dev_num).get_dataset()

        if self.non_iid_partitions > 0:
            # non-iid partitions
            self.train_indexes = self.model_loader.select_non_iid_samples(self.y_train, self.dev_num, self.data_sizes, self.non_iid_partitions)
            self.eval_indexes = self.model_loader.select_random_samples(self.y_test, self.dev_num, self.data_sizes)
        else:
            # random sampling
            self.train_indexes = self.model_loader.select_random_samples(self.y_train, self.dev_num, self.data_sizes)
            self.eval_indexes = self.model_loader.select_random_samples(self.y_test, self.dev_num, self.data_sizes)

    def start_worker(self, orchestrator_empty_queue):

        # register worker
        response = requests.post(self.orchestrator_address+"/register_worker")
        self.fillup_registration_fields(response.json())

        if response.status_code == 200:
            print("Registration was successful.")
            print("Init configuration: ", response.json())

            # handle available jobs
            while True:

                if orchestrator_empty_queue.value is False:
                    next_job = requests.get(self.orchestrator_address+"/get_next_jobs")

                    if len(next_job.text) > 0:

                        json_next_job = next_job.json()
                        print("Handle ", json_next_job["job_type"], " job ")
                        if json_next_job["job_type"] == 'fit':
                            self.handle_fit_job(json_next_job)
                        else:
                            self.handle_eval_job(json_next_job)
                    else:
                        sleep(1)

                else:
                    print("Supension because of empty jobs queue. ")
                sleep(1)
        else:
            print("Registration refused.")

    def load_local_data(self, fed_phase: FedPhase, dev_index: int):
        x_data = y_data = None
        if fed_phase == FedPhase.FIT:
            x_data = self.x_train[self.train_indexes[dev_index]]
            y_data = self.y_train[self.train_indexes[dev_index]]
        elif fed_phase == FedPhase.EVAL:
            x_data = self.x_test[self.eval_indexes[dev_index]]
            y_data = self.y_test[self.eval_indexes[dev_index]]
        return x_data, y_data

    def handle_fit_job(self, job):
        x_train, y_train = self.load_local_data(FedPhase.FIT, job["dev_index"])

        # run local data optimizer
        x_data, y_data = self.local_optimizer_fit.optimize(job["num_round"], job["dev_index"],
                                                                             job["num_examples"],
                                                                             self.dataset,
                                                                             self.dev_num,
                                                                             (x_train, y_train))

        model_factory = DatasetModelLoaderFactory.get_model_loader(self.dataset, self.dev_num)
        model = model_factory.get_compiled_model(optimizer=self.optimizer, metric=self.metric, train_data=(x_data, y_data))

        loss_func = model_factory.get_loss_function()
        global_weights = job["model_weights"]
        #if job["custom_loss"] is not None:
        #    loss_func = fedjob.custom_loss(model_factory.get_loss_function(), model, global_weights)

        model.compile(optimizer=tf.keras.optimizers.get(self.optimizer), run_eagerly=True, metrics=self.metric, loss=loss_func)

        if job["model_weights"] is not None:
            numpy_model_weights = []
            for x in job["model_weights"]:
                numpy_model_weights.append(numpy.array(x))
            model.set_weights(numpy_model_weights)

        # fit model
        history = model.fit(x_data, y_data, epochs=job["epochs"],
                                batch_size=job["batch_size"],
                                verbose=job["verbosity"])

        mean_metric = stats.mean(history.history[self.config.simulation["metric"]])
        mean_loss = stats.mean(history.history['loss'])
        model_weights = model.get_weights()
        json_model_weights = []
        for x in model_weights:
            json_model_weights.append(x.tolist())

        job_completed = {"mean_metric": mean_metric,
                         "mean_loss": mean_loss,
                         "model_weights": json_model_weights,
                         "num_examples": job["num_examples"],
                         "num_round": job["num_round"],
                         "dev_index": job["dev_index"],
                         "epochs": job["epochs"],
                         "batch_size": job["batch_size"]}


        requests.post(self.orchestrator_address + "/send_completed_job", json=job_completed)

    def handle_eval_job(self, job):
        x_train, y_train = self.load_local_data(FedPhase.EVAL, job["dev_index"])

        # run local data optimizer
        x_data, y_data = self.local_optimizer_fit.optimize(job["num_round"], job["dev_index"],
                                                           job["num_examples"],
                                                           self.dataset,
                                                           self.dev_num,
                                                           (x_train, y_train))

        model_factory = DatasetModelLoaderFactory.get_model_loader(self.dataset, self.dev_num)
        model = model_factory.get_compiled_model(optimizer=self.optimizer, metric=self.metric,
                                                 train_data=(x_data, y_data))

        loss_func = model_factory.get_loss_function()
        global_weights = job["model_weights"]
        # if job["custom_loss"] is not None:
        #    loss_func = fedjob.custom_loss(model_factory.get_loss_function(), model, global_weights)

        model.compile(optimizer=tf.keras.optimizers.get(self.optimizer), run_eagerly=True, metrics=self.metric,
                      loss=loss_func)

        #model.summary()

        if job["model_weights"] is not None:
            numpy_model_weights = []
            for x in job["model_weights"]:
                numpy_model_weights.append(numpy.array(x))
            model.set_weights(numpy_model_weights)

        loss, metric = model.evaluate(x_data, y_data, verbose=job["verbosity"])

        job_completed = {"metric": metric,
                         "loss": loss,
                         "num_examples": job["num_examples"],
                         "num_round": job["num_round"],
                         "dev_index": job["dev_index"],
                         "epochs": job["epochs"],
                         "batch_size": job["batch_size"]}

        requests.post(self.orchestrator_address + "/send_completed_job", json=job_completed)







