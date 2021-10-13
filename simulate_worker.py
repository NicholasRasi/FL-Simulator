import argparse
import ctypes
import logging
import multiprocessing
import os
from pathlib import Path
from random import randint
from fl_sim import Config
from fl_sim.worker.worker import Worker
from fl_sim.worker.worker_api import WorkerAPI


jobs_queue = multiprocessing.Queue()
orchestrator_empty_queue = multiprocessing.Value(ctypes.c_bool, False)

parser = argparse.ArgumentParser()
ROOT_DIR = Path(__file__).parent
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yml")
parser.add_argument('--config', default=CONFIG_PATH, type=str)
args = parser.parse_args()
logging.basicConfig(level='INFO')
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

config = Config(args.config)

ip_address = 'localhost'
port_number = randint(1000, 65535)

worker = multiprocessing.Process(target=Worker(ip_address, port_number, config, jobs_queue).start_worker, args=(orchestrator_empty_queue,))
worker.start()
api = multiprocessing.Process(target=WorkerAPI(port_number, orchestrator_empty_queue).start_api_listener, args=())
api.start()
worker.join()
api.join()
