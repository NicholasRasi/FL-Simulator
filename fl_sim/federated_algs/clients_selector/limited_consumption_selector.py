from typing import List
import numpy as np
from pulp import *
from fl_sim.federated_algs.clients_selector.clients_selector import ClientsSelector
from fl_sim.status.orchestrator_status import OrchestratorStatus


class LimitedConsumptionSelector(ClientsSelector):

    def __init__(self, config, status: OrchestratorStatus, logger, params=None):
        super().__init__(config, status, logger, params)

        # Parameters representing budget consumption (None is there is no limit)
        self.energy_limit = 4000
        self.resources_limit = 150
        self.network_limit = 16000000

    def select_devices(self, num_round: int) -> List:
        avail_indexes = self.get_available_devices(num_round)
        num_devs = int(self.config.algorithms["fit"]["params"]["k"] * avail_indexes.shape[0])

        # For the first round there is no history data so extract randomly num_devs devices
        if num_round == 0:
            dev_indexes = np.random.choice(avail_indexes, size=num_devs, replace=False)
        # For subsequent rounds
        else:
            # 1. Compute scores for each device depending on previous round loss
            scores = np.array([0 if len(x[x < sys.float_info.max]) == 0 else 1/x[
                        np.where(x != sys.float_info.max)[0][-1]] for x in
                              np.transpose(self.status.var["fit"]["model_metrics"]["loss"])])

            # 2. Get consumption estimation for each device
            resources_consumption = self.get_resources_consumption(num_round)
            network_consumption = self.get_network_consumption(num_round)
            energy_consumption = self.get_estimated_energy_consumption(num_round)

            # 3. Build optimization model
            keys = [str("Device_" + str(i)) for i in avail_indexes]
            values = [i for i in avail_indexes]
            dictionary = dict(zip(keys, values))

            model = LpProblem("Devices Selection", LpMinimize)
            device_vars = LpVariable.dicts("Device", avail_indexes, 0, cat='Binary')
            model += lpSum([scores[i] * device_vars[i] for i in avail_indexes])
            if self.energy_limit is not None:
                model += lpSum([energy_consumption[i] * device_vars[i] for i in avail_indexes]) <= self.energy_limit
            if self.resources_limit is not None:
                model += lpSum([resources_consumption[i] * device_vars[i] for i in avail_indexes]) <= self.resources_limit
            if self.network_limit is not None:
                model += lpSum([network_consumption * device_vars[i] for i in avail_indexes]) <= self.network_limit
            model += lpSum([device_vars[i] for i in avail_indexes]) == num_devs

            # 4. Solve optimization model
            model.solve(PULP_CBC_CMD(msg=False))

            # 5. Select devices chosen by the optimization solution
            dev_indexes = []
            for var in model.variables():
                if var.value() == 1:
                    dev_indexes.append(dictionary[var.name])
            dev_indexes = np.array(dev_indexes)

        return dev_indexes
