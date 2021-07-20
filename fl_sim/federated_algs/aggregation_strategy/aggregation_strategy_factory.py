from fl_sim import Status
from fl_sim.configuration import Config
from .fedavg_agg import FedAvgAgg
from .aggregation_strategy import AggregationStrategy


class AggregationStrategyFactory:

    @staticmethod
    def get_aggregation_strategy(selector: str, status: Status, data, config: Config, logger):
        if selector == "fedavg":
            return FedAvgAgg(status, data, config, logger)
        else:
            print("aia")
