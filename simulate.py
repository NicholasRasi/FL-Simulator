import logging
import argparse
import json
import coloredlogs
from fl_sim import Config
from fl_sim.simulator.sim import Simulator

LOG_FORMAT = "%(asctime)s:%(hostname)s:%(message)s"
logging.basicConfig(level='DEBUG', format=LOG_FORMAT)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt=LOG_FORMAT, datefmt="%H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config.yml", type=str)
args = parser.parse_args()

logging.info("init configuration...")
config = Config(args.config)
logging.info(json.dumps(config.__dict__, indent=3))

simulator = Simulator(config=config, logger=logger)
simulator.run()
