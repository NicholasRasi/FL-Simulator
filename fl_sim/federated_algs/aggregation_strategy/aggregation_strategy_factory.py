from fl_sim.configuration import Config
from .fedavg_agg import FedAvgAgg
from .fednova_agg import FedNovaAgg
from ...status.orchestrator_status import OrchestratorStatus


class AggregationStrategyFactory:

    @staticmethod
    def get_aggregation_strategy(selector: str, status: OrchestratorStatus, config: Config, logger):
        if selector == "fedavg":
            return FedAvgAgg(status, config, logger)
        elif selector == "fednova":
            return FedNovaAgg(status, config, logger)

