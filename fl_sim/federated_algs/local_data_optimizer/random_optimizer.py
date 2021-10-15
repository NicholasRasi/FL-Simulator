from .local_data_optimizer import LocalDataOptimizer
from fl_sim.dataset.model_loader_factory import DatasetModelLoaderFactory
from ...utils import FedPhase


class RandomOptimizer(LocalDataOptimizer):

    def optimize(self, num_round: int, dev_index: int, num_examples: int, model_name, num: int, data, fed_phase: FedPhase) -> tuple:
        x, y = data
        # get randomly num_examples examples among the available ones
        num_examples = min(x.shape[0], num_examples)
        sub_indexes = DatasetModelLoaderFactory.get_model_loader(model_name, num).select_random_samples(y, 1, [num_examples], fed_phase)[0]

        return x[sub_indexes], y[sub_indexes]
