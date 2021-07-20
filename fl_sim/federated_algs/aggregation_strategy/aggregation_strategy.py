from abc import ABC, abstractmethod
import numpy as np
from fl_sim import Status
from fl_sim.configuration import Config
from typing import List, Tuple

NDArrayList = List[np.ndarray]


class AggregationStrategy(ABC):

    def __init__(self, status: Status, data, config: Config, logger):
        self.config = config
        self.data = data
        self.status = status
        self.logger = logger

    @abstractmethod
    def aggregate_fit(weights: List[Tuple[int, NDArrayList]]) -> NDArrayList:
        """"
        Aggregation algorithm during fit
        """
    @abstractmethod
    def aggregate_losses(losses: List[Tuple[int, float]]) -> float:
        """"
        Aggregation of losses
        """

    @abstractmethod
    def aggregate_accuracies(accuracies: List[Tuple[int, float]]) -> float:
        """"
        Aggregation of accuracies
        """

    @abstractmethod
    def aggregate_evaluate(results: List[Tuple[int, float, float]]) -> Tuple[float, float]:
        """"
        Aggregation algorithm during evaluation
        """