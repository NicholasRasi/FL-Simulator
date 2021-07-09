from typing import List, Tuple
from .aggregation_strategy import AggregationStrategy
from .aggregate import Aggregate
from .aggregation_strategy import NDArrayList


class FedAvgAgg(AggregationStrategy):

    @staticmethod
    def aggregate_fit(weights: List[Tuple[int, NDArrayList]]) -> NDArrayList:
        return Aggregate.aggregate(weights)

    @staticmethod
    def aggregate_losses(losses: List[Tuple[int, float]]) -> float:
        return Aggregate.weighted_loss_avg(losses)

    @staticmethod
    def aggregate_accuracies(accuracies: List[Tuple[int, float]]) -> float:
        return Aggregate.weighted_accuracies_avg(accuracies)

    @staticmethod
    def aggregate_evaluate(results: List[Tuple[int, float, float]]) -> Tuple[float, float]:
        losses = [(num_samples, loss) for num_samples, loss, _ in results]
        accuracies = [(num_samples, acc) for num_samples, _, acc in results]
        return Aggregate.weighted_loss_avg(losses), Aggregate.weighted_accuracies_avg(accuracies)
