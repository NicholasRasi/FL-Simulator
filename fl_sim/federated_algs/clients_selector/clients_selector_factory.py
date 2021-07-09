from fl_sim import Status
from fl_sim.configuration import Config
from .best_ips_selector import BestIPSSelector
from .random_selector import RandomSelector


class ClientsSelectorFactory:

    @staticmethod
    def get_clients_selector(selector: str, config: Config, status: Status, logger):
        if selector == "random":
            return RandomSelector(config, status, logger)
        elif selector == "best_ips":
            return BestIPSSelector(config, status, logger)
