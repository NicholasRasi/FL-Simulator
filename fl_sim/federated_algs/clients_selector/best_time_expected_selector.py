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
            # 2. The stability (variance) of speed
            # 3. How many times a device has been selected
            comm_distribution_times = np.transpose(self.status.var["fit"]["times"]["communication_distribution"])
            comm_upload_times = np.transpose(self.status.var["fit"]["times"]["communication_upload"])

            # version 1
            global_opt_configs = self.status.var["fit"]["upd_opt_configs"]["global"]
            current_local_iterations = global_opt_configs["epochs"][num_round] * \
                                       global_opt_configs["num_examples"][
                                           num_round] / global_opt_configs["batch_size"][num_round]
            expected_computation_times = np.divide(current_local_iterations, self.status.con["devs"]["ips"])
            comm_times = comm_upload_times + comm_distribution_times
            mean_square_comm_times = np.asarray([0 if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2))+comp for t, comp in zip(comm_times, expected_computation_times)])
            #times = np.asarray([comm + comp for comm, comp in zip(mean_square_comm_times, expected_computation_times)])
            #print(mean_square_comm_times)
            var_times = np.asarray([0 if len(t[t > 0]) == 0 else np.var(t[t > 0]) for t in mean_square_comm_times])

            times_selected = [np.count_nonzero(t) for t in comm_times]
            #print(times_selected)
            fastest_index = np.multiply(mean_square_comm_times, self.fastests_parameter)
            more_stable_index = np.multiply(var_times, self.more_stable_parameter)
            fairness_index = np.multiply(times_selected, self.fairness_parameter * np.average(mean_square_comm_times[mean_square_comm_times > 0]))
            indexes = fastest_index + more_stable_index + fairness_index
            best = [x for x in np.argsort(indexes) if x in avail_indexes]
            fastests = [x for x in np.argsort(mean_square_comm_times) if x in avail_indexes]
            more_stable = [x for x in np.argsort(var_times) if x in avail_indexes]
            least_selected = [x for x in np.argsort(times_selected) if x in avail_indexes]


            # version 2
            #computation_times = np.transpose(self.status.var["fit"]["times"]["computation"])
            #times = computation_times + comm_upload_times + comm_distribution_times

            # avg_times = np.asarray([0 if len(t[t > 0]) == 0 else sum(t) / np.count_nonzero(t) for t in times])
            #mean_square_times = np.asarray(
            #    [0 if len(t[t > 0]) == 0 else np.sqrt(np.mean(t[t > 0] ** 2)) for t in times])
            #var_times = np.asarray([0 if len(t[t > 0]) == 0 else np.var(t[t > 0]) for t in times])

            #times_selected = [np.count_nonzero(t) for t in times]

            #fastest_index = np.multiply(mean_square_times, self.fastests_parameter)
            #more_stable_index = np.multiply(var_times, self.more_stable_parameter)
            #fairness_index = np.multiply(times_selected,
            #                             self.fairness_parameter * np.average(mean_square_times[mean_square_times > 0]))
            #indexes = fastest_index + more_stable_index + fairness_index
            #fastests = [x for x in np.argsort(mean_square_times) if x in avail_indexes]
            #more_stable = [x for x in np.argsort(var_times) if x in avail_indexes]
            #least_selected = [x for x in np.argsort(times_selected) if x in avail_indexes]
            #best = [x for x in np.argsort(indexes) if x in avail_indexes]
            #print("version no local")
            #print("fast ", fastests)
            #print("fair ", least_selected)
            #print("stable ", more_stable)

            dev_indexes = best[:num_devs]

        return dev_indexes
