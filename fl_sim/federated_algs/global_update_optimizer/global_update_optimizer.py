from abc import abstractmethod, ABC
from fl_sim import Status


class GlobalUpdateOptimizer(ABC):

    def __init__(self, epochs: int, batch_size: int, num_examples: int, status: Status, logger):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.status = status
        self.logger = logger

    @abstractmethod
    def optimize(self, r: int, dev_index: int, phase: str) -> dict:
        """
        :return: a dict with the number of epochs, batch size, num examples
        """
