from abc import ABC, abstractmethod
import numpy as np
from fl_sim.configuration import Config
from typing import List, Tuple
from fl_sim.status.orchestrator_status import OrchestratorStatus

NDArrayList = List[np.ndarray]


class AggregationStrategy(ABC):

    def __init__(self, status: OrchestratorStatus, config: Config, logger):
        self.config = config
        self.status = status
        self.logger = logger

    @abstractmethod
    def aggregate_fit(self, weights: List[Tuple[int, NDArrayList]]) -> NDArrayList:
        """"
        Aggregation of weights during fit
        """
    @abstractmethod
    def aggregate_losses(self, losses: List[Tuple[int, float]]) -> float:
        """"
        Aggregation of losses
        """

    @abstractmethod
    def aggregate_accuracies(self, accuracies: List[Tuple[int, float]]) -> float:
        """"
        Aggregation of accuracies
        """