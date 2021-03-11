import coloredlogs
import json
import logging
from config import Config
from fl_sim import Status, FedAvg

log_format = "%(asctime)s:%(hostname)s:%(message)s"
logging.basicConfig(level='DEBUG', format=log_format)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt=log_format, datefmt="%H:%M:%S")

logging.info("init configuration...")
config = Config()
logging.info(json.dumps(config.__dict__, indent=3))

logging.info("init status...")
init_status = Status(logger=logger, config=config)
fl = FedAvg(status=init_status, config=config, logger=logger)

logging.info("starting training...")
fl.run_server()

