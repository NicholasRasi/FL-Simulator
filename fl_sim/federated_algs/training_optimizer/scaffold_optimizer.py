from tensorflow import keras
from tensorflow.python.ops import state_ops


class SCAFFOLD_optimizer(keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, global_control_variate=0, local_control_variate=0, name="SCAFFOLDoptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._lr = learning_rate
        self.global_control_variate = global_control_variate
        self.local_control_variate = local_control_variate
        self._is_first = True

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv")
        for var in var_list:
            self.add_slot(var, "pg")

    def _resource_apply_dense(self, grad, var):
        print(self.local_control_variate)
        print(self.global_control_variate)
        var_update = state_ops.assign_sub(var, (self._lr * grad) - self.local_control_variate + self.global_control_variate)

        return var_update

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate")}
