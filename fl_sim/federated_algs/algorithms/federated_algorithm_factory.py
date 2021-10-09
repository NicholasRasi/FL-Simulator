from .fedavg import FedAvg
from .fedprox import FedProx


class FederatedAlgorithmFactory:

    @staticmethod
    def get_federated_algorithm(algorithm_name: str, status, config, logger):
        if algorithm_name == "fedavg":
            return FedAvg(status, config, logger)
        elif algorithm_name == "fedprox":
            return FedProx(status, config, logger)
        else:
            return None
