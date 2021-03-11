from .local_data_optimizer import LocalDataOptimizer
from ..status.dataset_model_loader import DatasetModelLoader


class RandomOptimizer(LocalDataOptimizer):

    def optimize(self, r: int, dev_index: int, num_examples: int, data) -> tuple:
        x, y = data
        # get randomly num_examples examples among the available ones
        num_examples = min(x.shape[0], num_examples)
        x_sub, y_sub = DatasetModelLoader.select_random_samples(x, y, num_examples)

        return x_sub, y_sub
