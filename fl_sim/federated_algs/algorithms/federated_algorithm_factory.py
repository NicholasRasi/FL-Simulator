from .fedavg import FedAvg
from .fedprox import FedProx


class FederatedAlgorithmFactory:

    @staticmethod
    def get_federated_algorithm(algorithm_name: str, status, config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock):
        if algorithm_name == "fedavg":
            return FedAvg(status, config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock)
        elif algorithm_name == "fedprox":
            return FedProx(status, config, logger, jobs_queue, completed_jobs_queue, workers_queue, lock)
        else:
            return None
