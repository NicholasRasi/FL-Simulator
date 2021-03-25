from .local_data_optimizer import LocalDataOptimizer
from ..status.dataset_model_loader import DatasetModelLoader


class RandomOptimizer(LocalDataOptimizer):

    def optimize(self, r: int, dev_index: int, num_examples: int, data) -> tuple:
        x, y = data
        # get randomly num_examples examples among the available ones
        num_examples = min(x.shape[0], num_examples)
        sub_indexes = DatasetModelLoader.select_random_samples(y, 1, [num_examples])[0]

        return x[sub_indexes], y[sub_indexes]
