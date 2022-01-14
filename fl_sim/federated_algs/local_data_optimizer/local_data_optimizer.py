from abc import abstractmethod, ABC
from fl_sim.utils import FedPhase


class LocalDataOptimizer(ABC):

    @abstractmethod
    def optimize(self, num_round: int, dev_index: int, num_examples: int, model_name, num: int, data, fed_phase: FedPhase) -> tuple:
        """
        :return: a subset data of size min(num_examples, available_examples), computed from the local data
        """
