import sys
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class BudgetedTimeRotationSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)
        self.time_desired = 7.2
        self.loss_frequency = 4
        self.fairness_frequency = 1
        self.fairness_counter = 0
        self.best_loss_counter = 0

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # For the first round there is no history data so extract randomly num_devs devices
        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:

            # Compute current mean round time
            current_mean_round_time = 0
            for round in range(num_round):
                print("round ", round)
                computation = self.status.var["fit"]["times"]["computation"][round]
                upload = self.status.var["fit"]["times"]["communication_upload"][round]
                distribution = self.status.var["fit"]["times"]["communication_distribution"][round]
                total_times = computation + upload + distribution
                current_mean_round_time += max(total_times)

            current_mean_round_time /= num_round

            # If the current average of round time is worse than the desired one then perform selection based on
            # fastests devices
            if current_mean_round_time > self.time_desired:
                print("fast")
                comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
                comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])
                global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
                current_local_iterations = global_opt_configs["epochs"][num_round] * \
                                           global_opt_configs["num_examples"][
                                               num_round] / global_opt_configs["batch_size"][num_round]
                expected_computation_times = np.divide(current_local_iterations, self.status.con["devs"]["ips"])
                comm_times = comm_upload_times + comm_distribution_times
                mean_square_comm_times = np.asarray(
                    [0 if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) for t in comm_times])
                total_times = np.asarray(
                    [comm + comp for comm, comp in zip(mean_square_comm_times, expected_computation_times)])
                fastests = [x for x in np.argsort(total_times) if x in avail_indexes]
                dev_indexes = fastests[:num_devs]
            # Otherwise, alternate rounds with least selected devices and rounds with biggest loss devices
            # accordingly to the frequencies
            else:
                if int(self.best_loss_counter/self.loss_frequency) <= int(self.fairness_counter/self.fairness_frequency):
                    print("loss")
                    # Select fastests devices
                    self.best_loss_counter += 1
                    losses = [sys.float_info.max if len(x[x < sys.float_info.max]) == 0 else x[
                        np.where(x != sys.float_info.max)[0][-1]] for x in
                              np.transpose(self.status.var["fit"]["model_metrics"]["loss"])]
                    biggest_loss = [x for x in np.argsort(losses) if x in avail_indexes]
                    dev_indexes = biggest_loss[-num_devs:]
                else:
                    print("fair")
                    # Select biggest loss devices
                    self.fairness_counter += 1
                    sel_by_dev = np.sum(self.status.var["fit"]["devs"]["selected"], axis=0)
                    least_selected = [x for x in np.argsort(sel_by_dev) if x in avail_indexes]
                    dev_indexes = least_selected[:num_devs]

        return dev_indexes