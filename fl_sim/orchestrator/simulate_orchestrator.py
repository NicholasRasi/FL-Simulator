import multiprocessing
from multiprocessing.managers import BaseManager
from pathlib import Path
from fl_sim.orchestrator import orchestrator_api
from fl_sim.orchestrator.orchestrator import Orchestrator
from fl_sim.orchestrator.workers_manager import WorkersManager
import os
import tensorflow as tf
import logging
import argparse
import json
import coloredlogs
from fl_sim import Config


if __name__ == '__main__':

    LOG_FORMAT = "%(asctime)s:%(hostname)s:%(message)s"
    logging.basicConfig(level='DEBUG', format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger, fmt=LOG_FORMAT, datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    ROOT_DIR = Path(__file__).parent.parent.parent
    CONFIG_PATH = os.path.join(ROOT_DIR, "config.yml")
    parser.add_argument('--config', default=CONFIG_PATH, type=str)
    args = parser.parse_args()

    logging.info("init configuration...")
    config = Config(args.config)
    logging.info(json.dumps(config.__dict__, indent=3))

    manager = multiprocessing.Manager()
    workers_queue = manager.list()
    jobs_queue = multiprocessing.Queue()
    completed_jobs_queue = multiprocessing.Queue()

    orchestrator = multiprocessing.Process(target=Orchestrator(jobs_queue, completed_jobs_queue, workers_queue, config, logger).start_orchestrator, args=())
    orchestrator.start()
    api = multiprocessing.Process(target=orchestrator_api.start_api_listener, args=(config, jobs_queue, completed_jobs_queue, workers_queue))
    api.start()
    orchestrator.join()
    api.join()

