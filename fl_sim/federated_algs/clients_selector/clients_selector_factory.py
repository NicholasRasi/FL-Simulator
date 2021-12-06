from fl_sim.configuration import Config
from .best_selector import BestSelector
from .best_time_expected_selector import BestTimeExpectedSelector
from .budgeted_fairness_rotation_selector import BudgetedFairnessRotationSelector
from .budgeted_time_rotation_selector import BudgetedTimeRotationSelector
from .dynamic_sampling import DynamicSamplingSelector
from .limited_consumption_selector import LimitedConsumptionSelector
from .limited_time_selector import LimitedTimeSelector
from .loss_and_fairness_selector import LossAndFairnessSelector
from .best_round_time_selector import BestRoundTimeSelector
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
        elif selector == "best_round_time":
            return BestRoundTimeSelector(config, status, logger)
        elif selector == "loss_and_fairness":
            return LossAndFairnessSelector(config, status, logger)
        elif selector == "best_time_expected":
            return BestTimeExpectedSelector(config, status, logger)
        elif selector == "budgeted_fairness_rotation":
            return BudgetedFairnessRotationSelector(config, status, logger)
        elif selector == "budgeted_time_rotation":
            return BudgetedTimeRotationSelector(config, status, logger)
        elif selector == "limited_time":
            return LimitedTimeSelector(config, status, logger)
        elif selector == "limited_consumption":
            return LimitedConsumptionSelector(config, status, logger)



