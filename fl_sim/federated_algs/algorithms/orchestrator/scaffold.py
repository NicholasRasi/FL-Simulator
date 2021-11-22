import requests
from json_tricks import loads, dumps
from fl_sim import FedAvg
from fl_sim.configuration import Config
from fl_sim.status.orchestrator_status import OrchestratorStatus
from fl_sim.utils import FedPhase


class SCAFFOLD(FedAvg):

    def __init__(self, status: OrchestratorStatus, config: Config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock):
        super().__init__(status, config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock)
        self.global_control_variate = 0
        self.local_control_variates = [0] * self.config.devices["num"]

    def put_client_job_fit(self, num_round: int, dev_index: int, global_config: dict):

        next_job = {"job_type": "fit",
                    "num_round": int(num_round),
                    "dev_index": int(dev_index),
                    "verbosity": self.config.simulation["tf_verbosity"],
                    "model_weights": self.status.global_model_weights,
                    "epochs": global_config["epochs"],
                    "batch_size": global_config["batch_size"],
                    "num_examples": global_config["num_examples"],
                    "local_control_variate": self.local_control_variates[int(dev_index)],
                    "global_control_variate": self.global_control_variate}

        self.lock.acquire()
        if self.jobs_queue.qsize() == 0:
            for worker in self.workers_queue:
                requests.post("http://" + worker[0] + ":" + str(worker[1]) + "/notify_available_jobs")
        self.jobs_queue.put(dumps(next_job))
        self.lock.release()

    def put_client_job_eval(self, num_round: int, dev_index: int, global_config: dict):

        next_job = {"job_type": "eval",
                    "num_round": int(num_round),
                    "dev_index": int(dev_index),
                    "verbosity": self.config.simulation["tf_verbosity"],
                    "model_weights": self.status.global_model_weights,
                    "epochs": global_config["epochs"],
                    "batch_size": global_config["batch_size"],
                    "local_control_variate": self.local_control_variates[int(dev_index)],
                    "global_control_variate": self.global_control_variate,
                    "num_examples": global_config["num_examples"]}

        self.lock.acquire()
        if self.jobs_queue.qsize() == 0:
            for worker in self.workers_queue:
                requests.post("http://" + worker[0] + ":" + str(worker[1]) + "/notify_available_jobs")
        self.jobs_queue.put(dumps(next_job))
        self.lock.release()

    def get_fit_results(self, completed_jobs_queue, created_jobs):
        fit_results = []
        local_delta_variates = []
        device_indexes = []
        for _ in range(created_jobs):
            fedres = loads(completed_jobs_queue.get())

            if self.status.con["model"]["tot_weights"] is None:
                self.status.con["model"]["tot_weights"] = sum([w_list.size for w_list in fedres.get("model_weights")])

            # update model weights with the new computed ones
            self.status.con["devs"]["local_models_weights"][fedres.get("dev_index")] = fedres.get("model_weights")

            # compute metrics
            local_iterations = fedres.get("epochs") * fedres.get("num_examples") / fedres.get("batch_size")
            computation_time = local_iterations / self.status.con["devs"]["ips"][fedres.get("dev_index")]
            network_consumption = 4 * self.status.con["model"]["tot_weights"]

            communication_time = network_consumption / \
                                 self.status.con["devs"]["net_speed"][fedres.get("num_round"), fedres.get("dev_index")]
            energy_consumption = self.config.energy["pow_comp_s"] * computation_time + \
                                 self.config.energy["pow_net_s"] * communication_time

            # update global configs status
            self.status.update_optimizer_configs(fedres.get("num_round"), fedres.get("dev_index"), FedPhase.FIT,
                                                 "local", fedres.get("epochs"), fedres.get("batch_size"),
                                                 fedres.get("num_examples"))

            local_delta_variates.append(fedres.get("local_control_delta"))
            device_indexes.append(fedres.get("dev_index"))

            # update status
            self.status.update_sim_data(fedres.get("num_round"), FedPhase.FIT, fedres.get("dev_index"),
                                        computation_time=computation_time,
                                        communication_time=communication_time,
                                        local_iterations=local_iterations,
                                        network_consumption=network_consumption,
                                        energy_consumption=energy_consumption,
                                        metric=fedres.get("mean_metric"),
                                        loss=fedres.get("mean_loss"))

            fit_results.append(
                (fedres.get("num_examples"), fedres.get("model_weights"), fedres.get("mean_loss"),
                 fedres.get("mean_metric")))

        self.compute_control_variates(local_delta_variates, device_indexes)

        return fit_results

    def compute_control_variates(self, local_delta_variates, device_indexes):
        global_delta_variates = sum(local_delta_variates) / len(local_delta_variates)

        self.global_control_variate = self.global_control_variate + (
                    (global_delta_variates * len(local_delta_variates)) / self.config.devices["num"])

        for i in range(len(local_delta_variates)):
            self.local_control_variates[device_indexes[i]] = self.local_control_variates[device_indexes[i]] + local_delta_variates[i]
