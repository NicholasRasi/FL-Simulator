from json_tricks import loads, dumps
import requests
from flask import Flask, request


def start_api_listener(config, status, jobs_queue, lock, completed_jobs_queue, workers_queue):
    app = FlaskAppWrapper('wrap', config, status, jobs_queue, lock, completed_jobs_queue, workers_queue)
    app.run(host="localhost", port=8000)


class EndpointAction(object):

    def __init__(self, action, config, status, jobs_queue, lock, completed_jobs_queue, workers_queue):
        self.action = action
        self.config = config
        self.status = status
        self.jobs_queue =jobs_queue
        self.lock = lock
        self.completed_jobs_queue = completed_jobs_queue
        self.workers_queue = workers_queue

    def __call__(self, *args):
        response = self.action(self.config, self.status, self.jobs_queue, self.lock, self.completed_jobs_queue, self.workers_queue)
        return response


class FlaskAppWrapper(object):

    def __init__(self, name, config, status, jobs_queue, lock, completed_jobs_queue, workers_queue):
        self.app = Flask(name)
        self.config = config
        self.status = status
        self.jobs_queue = jobs_queue
        self.lock = lock
        self.completed_jobs_queue = completed_jobs_queue
        self.workers_queue = workers_queue
        self.app.add_url_rule('/get_next_jobs', 'get_next_jobs', EndpointAction(get_next_jobs, self.config, self.status, jobs_queue, self.lock, completed_jobs_queue, workers_queue), methods=['GET'])
        self.app.add_url_rule('/register_worker', 'register_worker', EndpointAction(register_worker, self.config, self.status, jobs_queue, self.lock, completed_jobs_queue, workers_queue), methods=['GET', 'POST'])
        self.app.add_url_rule('/send_completed_job', 'send_completed_job', EndpointAction(send_completed_job, self.config, self.status, jobs_queue, self.lock, completed_jobs_queue, workers_queue), methods=['GET', 'POST'])

    def run(self, host, port):
        self.app.run(host=host, port=port)


def get_next_jobs(config, status, jobs_queue, lock, completed_jobs_queue, workers_queue):
    lock.acquire()
    if jobs_queue.qsize() > 0:
        lock.release()
        return jobs_queue.get()
    else:
        for worker in workers_queue:
            requests.post("http://" + worker[0] + ":" + str(worker[1]) + "/notify_empty_queue")
        lock.release()
    return "", 200


def register_worker(config, status, jobs_queue, lock, completed_jobs_queue, workers_queue):
    # register worker
    workers_queue.append((request.json["ip_address"], request.json["port_number"]))

    # send back initial configuration
    init_configuration = {"dataset": config.simulation["model_name"],
                           "dev_num": config.devices["num"],
                           "train_indexes": status.con["devs"]["local_data"][0],
                           "eval_indexes": status.con["devs"]["local_data"][1],
                           "fedalg": config.algorithms["federated_algorithm"],
                           "verbosity": config.simulation["tf_verbosity"],
                           "optimizer": config.algorithms["optimizer"],
                           "metric": config.simulation["metric"],
                           "local_optimizer_fit": config.algorithms["fit"]["data"],
                           "local_optimizer_eval": config.algorithms["eval"]["data"]
                          }
    return dumps(init_configuration), 200


def send_completed_job(config, status, jobs_queue, lock, completed_jobs_queue, workers_queue):
    completed_jobs_queue.put(request.json)
    return "", 200
