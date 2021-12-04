import sys
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class BudgetedTimeRotationSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)

        # Average round time desired
        self.time_desired = 7.2

        # Frequency in number of rounds with which perform selection based on loss
        self.loss_frequency = 4

        # Frequency in number of rounds with which perform selection based on least selected devices
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
                computation = self.status.var["fit"]["times"]["computation"][round]
                upload = self.status.var["fit"]["times"]["communication_upload"][round]
                distribution = self.status.var["fit"]["times"]["communication_distribution"][round]
                total_times = computation + upload + distribution
                current_mean_round_time += max(total_times)

            current_mean_round_time /= num_round

            # If the current mean round time is worse than the desired one then perform selection based on
            # fastest devices
            if current_mean_round_time > self.time_desired:
                mean_square_times = self.get_quadratic_mean_times(num_round)
                fastest = [x for x in np.argsort(mean_square_times) if x in avail_indexes]
                dev_indexes = fastest[:num_devs]
            # Otherwise, alternate rounds with least selected devices and rounds with biggest loss devices
            # accordingly to the frequencies
            else:
                if int(self.best_loss_counter/self.loss_frequency) <= int(self.fairness_counter/self.fairness_frequency):
                    # Select biggest loss devices
                    self.best_loss_counter += 1
                    losses = [sys.float_info.max if len(x[x < sys.float_info.max]) == 0 else x[
                        np.where(x != sys.float_info.max)[0][-1]] for x in
                              np.transpose(self.status.var["fit"]["model_metrics"]["loss"])]
                    biggest_loss = [x for x in np.argsort(losses) if x in avail_indexes]
                    dev_indexes = biggest_loss[-num_devs:]
                else:
                    # Select least selected devices
                    self.fairness_counter += 1
                    sel_by_dev = np.sum(self.status.var["fit"]["devs"]["selected"], axis=0)
                    least_selected = [x for x in np.argsort(sel_by_dev) if x in avail_indexes]
                    dev_indexes = least_selected[:num_devs]

        return dev_indexes