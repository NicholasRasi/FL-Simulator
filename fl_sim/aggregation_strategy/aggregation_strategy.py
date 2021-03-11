from abc import ABC
import numpy as np
from typing import List

NDArrayList = List[np.ndarray]


class AggregationStrategy(ABC):
    pass
