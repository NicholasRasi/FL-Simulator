from typing import List, Tuple

import numpy as np

from . import FedAvgAgg
from .aggregate import Aggregate
from .aggregation_strategy import NDArrayList


class FedDynAgg(FedAvgAgg):

    def aggregate_fit(self, weights: List[Tuple[NDArrayList]]) -> NDArrayList:
        return Aggregate.average(weights)

    def aggregate_losses(self, losses: List[Tuple[int, float]]) -> float:
        return Aggregate.weighted_loss_avg(losses)

    def aggregate_accuracies(self, accuracies: List[Tuple[int, float]]) -> float:
        return Aggregate.weighted_accuracies_avg(accuracies)

    def aggregate_evaluate(self, results: List[Tuple[int, float, float]]) -> Tuple[float, float]:
        losses = [(num_samples, loss) for num_samples, loss, _ in results]
        accuracies = [(num_samples, acc) for num_samples, _, acc in results]
        return Aggregate.weighted_loss_avg(losses), Aggregate.weighted_accuracies_avg(accuracies)
