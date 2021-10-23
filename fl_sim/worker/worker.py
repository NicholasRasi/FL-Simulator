import logging
from time import sleep
import requests
from json_tricks import loads
from fl_sim.federated_algs.algorithms.worker.worker_algorithm_factory import WorkerAlgorithmFactory


class Worker:
    def __init__(self, ip_address, port_number, config, jobs_queue):
        self.ip_address = ip_address
        self.port_number = port_number
        self.config = config
        self.jobs_queue = jobs_queue
        self.orchestrator_address = "http://127.0.0.1:8000"
        self.federated_worker = None

    def start_worker(self, orchestrator_empty_queue):
        # register worker
        response = requests.post(self.orchestrator_address + "/register_worker", json={"ip_address": self.ip_address, 'port_number': self.port_number})
        init_conf = loads(response.text)
        self.federated_worker = WorkerAlgorithmFactory.get_federated_algorithm(init_conf["fedalg"], self.ip_address, self.port_number, self.config, self.jobs_queue, init_conf)

        if response.status_code == 200:
            logging.info("Registration was successful.")
            del init_conf["train_indexes"]
            del init_conf["eval_indexes"]
            logging.info("Init configuration: " + str(init_conf))

            # handle available jobs
            while True:

                if orchestrator_empty_queue.value is False:
                    next_job = requests.get(self.orchestrator_address+"/get_next_jobs")

                    if len(next_job.text) > 0:

                        json_next_job = loads(next_job.text)
                        logging.info(str("Handle " + json_next_job["job_type"] + " job , num round " + str(json_next_job["num_round"]) + " dev index " + str(json_next_job["dev_index"]) + ", num examples " + str(json_next_job["num_examples"])))
                        if json_next_job["job_type"] == 'fit':
                            self.federated_worker.handle_fit_job(json_next_job)
                        else:
                            self.federated_worker.handle_eval_job(json_next_job)
                    else:
                        sleep(1)
                else:
                    sleep(1)
        else:
            logging.info("Registration refused.")
