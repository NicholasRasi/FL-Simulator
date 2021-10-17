from fl_sim.federated_algs.algorithms.worker.fedavg_worker import FedAvgWorker
from fl_sim.federated_algs.algorithms.worker.fedprox_worker import FedProxWorker
from fl_sim.federated_algs.algorithms.worker.scaffold_worker import SCAFFOLDWorker


class WorkerAlgorithmFactory:

    @staticmethod
    def get_federated_algorithm(algorithm_name: str, ip_address, port_number, config, jobs_queue, init_conf):
        if algorithm_name == "fedavg":
            return FedAvgWorker(ip_address, port_number, config, jobs_queue, init_conf)
        if algorithm_name == "fedprox":
            return FedProxWorker(ip_address, port_number, config, jobs_queue, init_conf)
        if algorithm_name == "fednova":
            return FedAvgWorker(ip_address, port_number, config, jobs_queue, init_conf)
        if algorithm_name == "scaffold":
            return SCAFFOLDWorker(ip_address, port_number, config, jobs_queue, init_conf)
        else:
            return None
