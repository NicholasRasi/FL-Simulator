import tensorflow as tf
import numpy as np


def feddyn_loss(loss_type, model, alfa_parameter, gradients, global_weights):

    def loss(y_true, y_pred):

        local_model_weights = model.weights

        # Task Loss
        loss_function = tf.keras.losses.get(loss_type)
        task_loss = loss_function(y_true, y_pred)

        # Linear penalty
        linear_penalty = 0
        if gradients is not 0:
            linear_penalty = local_model_weights * gradients

        # Quadratic penalty
        quadratic_penalty = 0
        if global_weights is not None:
            for index, (value1, value2) in enumerate(zip(local_model_weights, global_weights)):
                quadratic_penalty += ((alfa_parameter / 2) * np.linalg.norm(np.subtract(value1.numpy(), value2)) ** 2)

        loss_result = task_loss - linear_penalty + quadratic_penalty

        return loss_result

    return loss