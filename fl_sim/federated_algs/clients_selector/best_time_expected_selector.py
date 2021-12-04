from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class BestTimeExpectedSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)

        self.fairness_parameter = 0.33
        self.fastests_parameter = 0.34
        self.more_stable_parameter = 0.33
        assert self.fairness_parameter + self.fastests_parameter + self.more_stable_parameter == 1

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # For the first round there is no history data on times so extract randomly num_devs devices
        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        # For subsequent rounds
        else:
            # The selection depends on
            # 1. The average speed
            mean_square_times = self.get_quadratic_mean_times(num_round)
            fastest_index = np.multiply(mean_square_times, self.fastests_parameter)

            # 2. The stability (variance) of speed
            var_times = np.asarray([0 if len(t[t > 0]) == 0 else np.var(t[t > 0]) for t in mean_square_times])
            more_stable_index = np.multiply(var_times, self.more_stable_parameter)

            # 3. How many times a device has been selected
            times_selected = np.sum(self.status.var["fit"]["devs"]["selected"], axis=0)
            fairness_index = np.multiply(times_selected, self.fairness_parameter * np.average(mean_square_times[mean_square_times > 0]))

            # Obtain final score indexes as the sum of each term
            indexes = fastest_index + more_stable_index + fairness_index
            best = [x for x in np.argsort(indexes) if x in avail_indexes]

            # Select num_devs devices with highest score
            dev_indexes = best[:num_devs]

        return dev_indexes
