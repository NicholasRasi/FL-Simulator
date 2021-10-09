from random import randint

from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast


class WorkerAPI:
    def __init__(self, jobs_queue):
        self.jobs_queue = jobs_queue
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.api.add_resource(NotifyEmptyQueue, '/notify_empty_queue')
        self.api.add_resource(NotifyAvailableJobs, '/notify_available_jobs')

    def start_api_listener(self, orchestrator_empty_queue):
        self.app.run(host="localhost", port=randint(1000, 65535))


class NotifyEmptyQueue(Resource):
    def get(self):
        return {'data': "myData"}, 200  # return data and 200 OK code


class NotifyAvailableJobs(Resource):
    def post(self):
        return {'data': "myData"}, 200  # return data and 200 OK code

