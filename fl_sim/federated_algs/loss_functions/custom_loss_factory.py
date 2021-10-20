from fl_sim.federated_algs.loss_functions.fed_prox_loss import fed_prox_loss
from fl_sim.federated_algs.loss_functions.feddyn_loss import feddyn_loss


class CustomLossFactory:

    @staticmethod
    def get_custom_loss(loss_name: str):
        if loss_name == "fed_prox_loss":
            return fed_prox_loss
        elif loss_name == "feddyn_loss":
            return feddyn_loss
        else:
            return None
