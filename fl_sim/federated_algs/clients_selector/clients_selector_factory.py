from fl_sim import Status
from fl_sim.configuration import Config
from .best_selector import BestSelector
from .random_selector import RandomSelector


class ClientsSelectorFactory:

    @staticmethod
    def get_clients_selector(selector: str, config: Config, status: Status, logger):
        if selector == "random":
            return RandomSelector(config, status, logger)
        elif selector == "best_ips":
            return BestSelector(config, status, logger, params={"sortBy": "ips"})
        elif selector == "best_net_speed":
            return BestSelector(config, status, logger, params={"sortBy": "net_speed"})
        elif selector == "best_energy":
            return BestSelector(config, status, logger, params={"sortBy": "energy"})
