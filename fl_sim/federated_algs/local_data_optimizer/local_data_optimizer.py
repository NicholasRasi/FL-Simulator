from abc import abstractmethod, ABC
from fl_sim import Status


class LocalDataOptimizer(ABC):

    def __init__(self, status: Status, logger):
        self.status = status
        self.logger = logger

    @abstractmethod
    def optimize(self, r: int, dev_index: int, num_examples: int, data) -> tuple:
        """
        :return: a subset data of size min(num_examples, available_examples), computed from the local data
        """
