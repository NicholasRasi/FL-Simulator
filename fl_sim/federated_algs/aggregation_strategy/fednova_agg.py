from typing import List, Tuple
from . import FedAvgAgg
from .aggregate import Aggregate
from .aggregation_strategy import NDArrayList


class FedNovaAgg(FedAvgAgg):

    def aggregate_fit(self, weights: List[Tuple[int, NDArrayList, int]]) -> NDArrayList:
        return Aggregate.federated_normalized_averaging(self.status.global_model_weights, weights)
