from .global_update_optimizer import GlobalUpdateOptimizer


class StaticOptimizer(GlobalUpdateOptimizer):

    def optimize(self, r: int, dev_index: int) -> dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size, "num_examples": self.num_examples}
