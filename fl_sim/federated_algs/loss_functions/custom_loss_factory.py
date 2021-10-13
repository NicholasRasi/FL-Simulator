from fl_sim.federated_algs.loss_functions.fed_prox_loss import fed_prox_loss


class CustomLossFactory:

    @staticmethod
    def get_custom_loss(loss_name: str):
        if loss_name == "fed_prox_loss":
            return fed_prox_loss
        else:
            return None
