import coloredlogs
import json
import logging
from fl_sim import FedAvg, Config
import argparse

log_format = "%(asctime)s:%(hostname)s:%(message)s"
logging.basicConfig(level='DEBUG', format=log_format)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt=log_format, datefmt="%H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config.yml", type=str)
args = parser.parse_args()

logging.info("init configuration...")
config = Config(args.config)
logging.info(json.dumps(config.__dict__, indent=3))

fl = FedAvg(config=config, logger=logger)
fl.run_server()

