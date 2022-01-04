from typing import List, Tuple
from .aggregation_strategy import AggregationStrategy
from .aggregate import Aggregate
from .aggregation_strategy import NDArrayList


class FedAvgAgg(AggregationStrategy):

    def aggregate_fit(self, weights: List[Tuple[int, NDArrayList]]) -> NDArrayList:
        return Aggregate.aggregate(weights)

    def aggregate_losses(self, losses: List[Tuple[int, float]]) -> float:
        return Aggregate.weighted_loss_avg(losses)

    def aggregate_accuracies(self, accuracies: List[Tuple[int, float]]) -> float:
        return Aggregate.weighted_accuracies_avg(accuracies)

