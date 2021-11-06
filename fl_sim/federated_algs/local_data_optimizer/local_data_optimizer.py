from abc import abstractmethod, ABC


class LocalDataOptimizer(ABC):

    @abstractmethod
    def optimize(self, num_round: int, dev_index: int, num_examples: int, data) -> tuple:
        """
        :return: a subset data of size min(num_examples, available_examples), computed from the local data
        """
