import tensorflow as tf
import numpy as np


def fed_prox_loss(loss_type, model, global_weights):

    def loss(y_true, y_pred):

        # hyper parameter to be properly tuned
        mu_parameter = 0.05

        # get loss without fedprox regularization term
        loss_function = tf.keras.losses.get(loss_type)
        loss_result = loss_function(y_true, y_pred)

        local_model_weights = model.weights
        global_model_weights = global_weights
        if global_model_weights is None:
            return loss_result

        # compute fedprox regularization term
        fed_prox_reg = 0
        for index, (value1, value2) in enumerate(zip(local_model_weights, global_model_weights)):
            fed_prox_reg += ((mu_parameter / 2) * np.linalg.norm(np.subtract(value1.numpy(), value2)) ** 2)

        # add fedprox regularization term to the loss
        loss_result = loss_result + fed_prox_reg

        return loss_result

    return loss