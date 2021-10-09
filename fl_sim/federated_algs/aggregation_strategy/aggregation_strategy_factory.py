from fl_sim.configuration import Config
from .fedavg_agg import FedAvgAgg
from ...status.orchestrator_status import OrchestratorStatus


class AggregationStrategyFactory:

    @staticmethod
    def get_aggregation_strategy(selector: str, status: OrchestratorStatus, data, config: Config, logger):
        if selector == "fedavg":
            return FedAvgAgg(status, data, config, logger)

