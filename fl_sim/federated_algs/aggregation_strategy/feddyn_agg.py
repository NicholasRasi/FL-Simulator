from typing import List, Tuple
from . import FedAvgAgg
from .aggregate import Aggregate
from .aggregation_strategy import NDArrayList
from ... import Config
import numpy as np
from ...status.orchestrator_status import OrchestratorStatus


class FedDynAgg(FedAvgAgg):

    def __init__(self, status: OrchestratorStatus, config: Config, logger, h_parameter, alfa_parameter):
        super().__init__(status, config, logger)
        self.h_parameter = h_parameter
        self.alfa_parameter = alfa_parameter

    def aggregate_fit(self, weights: List[Tuple[NDArrayList]]) -> NDArrayList:
        return np.subtract(Aggregate.average(weights), (1/self.alfa_parameter) * self.h_parameter)

