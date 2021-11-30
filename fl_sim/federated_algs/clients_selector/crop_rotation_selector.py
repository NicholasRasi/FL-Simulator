import sys
from typing import List
import numpy as np
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class CropRotationSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)
        self.fairness_desired = 1.5
        self.best_time_selection = 0
        self.best_loss_selection = 0

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # For the first round there is no history data so extract randomly num_devs devices
        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        else:

            # Compute current fairness
            sel_by_dev = np.sum(self.status.var["fit"]["devs"]["selected"], axis=0)
            current_fairness = np.std(sel_by_dev)

            # If the current fairness is worse than the desired one then perform selection based on
            # least selected devices
            if current_fairness > self.fairness_desired:
                least_selected = [x for x in np.argsort(sel_by_dev) if x in avail_indexes]
                dev_indexes = least_selected[:num_devs]
            # Otherwise, alternate one round with fastest devices and one round with biggest loss devices
            else:
                if self.best_time_selection <= self.best_loss_selection:
                    # Select fastests devices
                    self.best_time_selection += 1
                    comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
                    comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])
                    computation_times = np.transpose(self.status.var["fit"]["times"]["computation"])

                    times = computation_times + comm_upload_times + comm_distribution_times

                    mean_square_times = np.asarray([0 if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) for t in times])
                    fastests = [x for x in np.argsort(mean_square_times) if x in avail_indexes]
                    dev_indexes = fastests[:num_devs]
                else:
                    # Select biggest loss devices
                    self.best_loss_selection += 1
                    losses = [sys.float_info.max if len(x[x < sys.float_info.max]) == 0 else x[np.where(x != sys.float_info.max)[0][-1]] for x in np.transpose(self.status.var["fit"]["model_metrics"]["loss"])]
                    biggest_loss = [x for x in np.argsort(losses) if x in avail_indexes]
                    print(biggest_loss)
                    dev_indexes = biggest_loss[-num_devs:]
                    print("selezionati ", dev_indexes)

        return dev_indexes
