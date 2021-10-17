import tensorflow as tf
from fl_sim.federated_algs.training_optimizer.scaffold_optimizer import SCAFFOLD_optimizer


class TrainingOptimizerFactory:

    @staticmethod
    def get_training_optimizer(selector: str, job):
        if selector == "SCAFFOLD":
            return SCAFFOLD_optimizer(job)
        else:
            return tf.keras.optimizers.get(selector)
