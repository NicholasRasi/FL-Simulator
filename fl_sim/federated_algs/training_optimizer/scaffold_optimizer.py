import numpy as np
from tensorflow import keras
from tensorflow.python.ops import state_ops


class SCAFFOLD_optimizer(keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, global_control_variate=0, local_control_variate=0, name="SCAFFOLDoptimizer", num_layers=0, **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._lr = learning_rate
        self.global_control_variate = global_control_variate
        self.local_control_variate = local_control_variate
        self._is_first = True
        self.num_layers = num_layers
        self.current_index = 0

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv")
        for var in var_list:
            self.add_slot(var, "pg")

    def _resource_apply_dense(self, grad, var):
        x = grad

        if self.local_control_variate is not 0:
            control_variates_diff = np.subtract(self.global_control_variate[self.current_index % self.num_layers], self.local_control_variate[self.current_index % self.num_layers])
            x = x + control_variates_diff

        x = self._lr * x
        var_update = state_ops.assign_sub(var, x)
        self.current_index = self.current_index + 1

        return var_update

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        x = grad

        if self.local_control_variate is not 0:
            control_variates_diff = np.subtract(self.global_control_variate[self.current_index % self.num_layers],
                                                self.local_control_variate[self.current_index % self.num_layers])
            x = x + control_variates_diff

        x = self._lr * x
        var_update = state_ops.assign_sub(var, x)
        #print(var.shape)
        #print(x.shape)
        #var_update = var
        self.current_index = self.current_index + 1

        return var_update

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate")}
