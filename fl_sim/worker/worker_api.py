import ctypes
from multiprocessing import Value
from flask import Flask
from flask_restful import Api


class EndpointAction(object):

    def __init__(self, action, orchestrator_empty_queue):
        self.action = action
        self.orchestrator_empty_queue = orchestrator_empty_queue

    def __call__(self, *args):
        response = self.action(self.orchestrator_empty_queue)
        return response


class WorkerAPI:
    def __init__(self, port_number, orchestrator_empty_queue):
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.port_number = port_number
        self.orchestrator_empty_queue = orchestrator_empty_queue
        self.app.add_url_rule('/notify_empty_queue', 'notify_empty_queue', EndpointAction(notify_empty_queue, self.orchestrator_empty_queue), methods=['GET', 'POST'])
        self.app.add_url_rule('/notify_available_jobs', 'notify_available_jobs', EndpointAction(notify_available_jobs, self.orchestrator_empty_queue), methods=['GET', 'POST'])

    def start_api_listener(self):
        self.app.run(host='localhost', port=self.port_number)


def notify_empty_queue(orchestrator_empty_queue):
    orchestrator_empty_queue.value = Value(ctypes.c_bool, True)
    return "", 200


def notify_available_jobs(orchestrator_empty_queue):
    orchestrator_empty_queue.value = not Value(ctypes.c_bool, True)  # Value(ctypes.c_bool, False) not working
    return "", 200


