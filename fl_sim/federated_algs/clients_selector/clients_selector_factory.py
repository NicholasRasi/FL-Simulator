from fl_sim.configuration import Config
from .best_selector import BestSelector
from .dynamic_sampling import DynamicSamplingSelector
from .exploration_exploitation_selector import ExplorationExploitationSelector
from .fedcs_selector import FedCSSelector
from .random_selector import RandomSelector
from ...status.orchestrator_status import OrchestratorStatus


class ClientsSelectorFactory:

    @staticmethod
    def get_clients_selector(selector: str, config: Config, status: OrchestratorStatus, logger):
        if selector == "random":
            return RandomSelector(config, status, logger)
        elif selector == "best_ips":
            return BestSelector(config, status, logger, params={"sortBy": "ips"})
        elif selector == "best_net_speed":
            return BestSelector(config, status, logger, params={"sortBy": "net_speed"})
        elif selector == "best_energy":
            return BestSelector(config, status, logger, params={"sortBy": "energy"})
        elif selector == "dynamic_sampling":
            return DynamicSamplingSelector(config, status, logger)
        elif selector == "fedcs":
            return FedCSSelector(config, status, logger)
        elif selector == "exploration_exploitation":
            return ExplorationExploitationSelector(config, status, logger)
