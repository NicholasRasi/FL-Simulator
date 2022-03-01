import random
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
        self.auto_tuning = True

        # Frequency in number of rounds with which perform selection based on loss
        self.loss_frequency = 1

        # Frequency in number of rounds with which perform selection based on least selected devices
        self.fairness_frequency = 50

        self.fairness_counter = 0
        self.best_loss_counter = 0

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # For the first round there is no history data so extract randomly num_devs devices
        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:

            if self.auto_tuning:
                self.update_time_desired(num_round)

            # Compute current mean round time
            rt = self.status.var["fit"]["times"]["computation"] + self.status.var["fit"]["times"]["communication_upload"] + \
                 self.status.var["fit"]["times"]["communication_distribution"]
            max_round = np.amax(rt, axis=1)
            current_mean_round_time = np.mean(max_round[:num_round])

            # If the current mean round time is worse than the desired one then perform selection based on
            # fastest devices
            if current_mean_round_time > self.time_desired:

                self.is_last_fast = True

                mean_square_times = self.get_quadratic_mean_times(num_round)
                fastest = [x for x in np.argsort(mean_square_times) if x in avail_indexes]
                dev_indexes = fastest[:num_devs]
            # Otherwise, alternate rounds with least selected devices and rounds with biggest loss devices
            # accordingly to the frequencies
            else:

                self.is_last_fast = False

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

    def update_time_desired(self, num_round):

        previous_comp = self.status.var["fit"]["times"]["computation"][num_round - 1]
        previous_upload = self.status.var["fit"]["times"]["communication_upload"][num_round - 1]
        previous_distr = self.status.var["fit"]["times"]["communication_distribution"][num_round - 1]
        previous_time = max(previous_distr + previous_upload + previous_comp)

        # If second round initialize variables
        if num_round == 1:
            self.time_desired = previous_time
            self.is_last_fast = False
            self.fastest_mean = []
            self.not_fastest_mean = []
            self.not_fastest_mean.append(previous_time)
        # Otherwise update desired time
        else:
            if self.is_last_fast:
                self.fastest_mean.append(previous_time)
            else:
                self.not_fastest_mean.append(previous_time)

            fast_m = 0
            if len(self.fastest_mean) > 0:
                fast_m = sum(self.fastest_mean) / len(self.fastest_mean)

            not_fast_m = 0
            if len(self.not_fastest_mean) > 0:
                not_fast_m = sum(self.not_fastest_mean) / len(self.not_fastest_mean)

            self.time_desired = random.uniform(fast_m, not_fast_m)
