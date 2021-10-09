import json
import multiprocessing
from time import time, sleep

from flask import Flask, Response, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast


def start_api_listener(config, jobs_queue, completed_jobs_queue, workers_queue):
    app = FlaskAppWrapper('wrap', config, jobs_queue, completed_jobs_queue, workers_queue)
    app.run(host="localhost", port=8000)


class EndpointAction(object):

    def __init__(self, action, config, jobs_queue, completed_jobs_queue, workers_queue):
        self.action = action
        self.config = config
        self.jobs_queue =jobs_queue
        self.completed_jobs_queue = completed_jobs_queue
        self.workers_queue = workers_queue

    def __call__(self, *args):
        response = self.action(self.config, self.jobs_queue, self.completed_jobs_queue, self.workers_queue)
        return response


class FlaskAppWrapper(object):

    def __init__(self, name, config, jobs_queue, completed_jobs_queue, workers_queue):
        self.app = Flask(name)
        self.config = config
        self.jobs_queue = jobs_queue
        self.completed_jobs_queue = completed_jobs_queue
        self.workers_queue = workers_queue
        self.app.add_url_rule('/get_next_jobs', 'get_next_jobs', EndpointAction(get_next_jobs, self.config, jobs_queue, completed_jobs_queue, workers_queue), methods=['GET'])
        self.app.add_url_rule('/register_worker', 'register_worker', EndpointAction(register_worker, self.config, jobs_queue, completed_jobs_queue, workers_queue), methods=['GET', 'POST'])
        self.app.add_url_rule('/send_completed_job', 'send_completed_job', EndpointAction(send_completed_job, self.config, jobs_queue, completed_jobs_queue, workers_queue), methods=['GET', 'POST'])

    def run(self, host, port):
        self.app.run(host=host, port=port)


def get_next_jobs(config, jobs_queue, completed_jobs_queue, workers_queue):
    if jobs_queue.qsize() > 0:
        return jobs_queue.get()
    return "", 200


def register_worker(config, jobs_queue, completed_jobs_queue, workers_queue):

    # register worker
    workers_queue.append((request.environ['REMOTE_ADDR'], request.environ['REMOTE_PORT']))

    # send back initial configuration
    init_configuration = {"dataset": config.simulation["model_name"],
                           "dev_num": config.devices["num"],
                           "fedalg": config.algorithms["federated_algorithm"],
                           "verbosity": config.simulation["tf_verbosity"],
                           "optimizer": config.algorithms["optimizer"],
                           "metric": config.simulation["metric"],
                           "non_iid_partitions": config.data["non_iid_partitions"],
                           "local_optimizer_fit": config.algorithms["fit"]["data"],
                           "local_optimizer_eval": config.algorithms["eval"]["data"]
                          }
    return json.dumps(init_configuration), 200


def send_completed_job(config, jobs_queue, completed_jobs_queue, workers_queue):
    completed_jobs_queue.put(request.get_json(force=True))
    return "", 200
