from config import Config
from fl_sim import Status
from .best_rt_selector import BestRTSelector
from .random_selector import RandomSelector


class ClientsSelectorFactory:

    @staticmethod
    def get_clients_selector(selector: str, config: Config, status: Status, logger):
        if selector == "random":
            return RandomSelector(config, status, logger)
        elif selector == "best_ips":
            return BestRTSelector(config, status, logger)
