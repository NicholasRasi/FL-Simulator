import argparse
import ctypes
import multiprocessing
import os
from pathlib import Path
from fl_sim import Config
from fl_sim.worker.worker import Worker
from fl_sim.worker.worker_api import WorkerAPI


jobs_queue = multiprocessing.Queue()
orchestrator_empty_queue = multiprocessing.Value(ctypes.c_bool, False)

parser = argparse.ArgumentParser()
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yml")
parser.add_argument('--config', default=CONFIG_PATH, type=str)
args = parser.parse_args()
config = Config(args.config)

worker = multiprocessing.Process(target=Worker(config, jobs_queue).start_worker, args=(orchestrator_empty_queue,))
worker.start()
api = multiprocessing.Process(target=WorkerAPI(jobs_queue).start_api_listener, args=(orchestrator_empty_queue,))
api.start()

