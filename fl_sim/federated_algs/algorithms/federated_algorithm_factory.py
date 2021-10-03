from .fedavg import FedAvg
from .fedprox import FedProx


class FederatedAlgorithmFactory:

    @staticmethod
    def get_federated_algorithm(algorithm_name: str, status, data, config, logger):
        if algorithm_name == "fedavg":
            return FedAvg(status, data, config, logger)
        elif algorithm_name == "fedprox":
            return FedProx(status, data, config, logger)
        else:
            return None
