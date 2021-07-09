import numpy as np
from abc import ABC
from fl_sim import Status
from fl_sim.configuration import Config


class FedAlg(ABC):

    def __init__(self, status: Status, data, config: Config, logger):
        self.status = status
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.config = config
        self.logger = logger

    def get_failed_devs(self, num_round: int):
        return np.where(self.status.con["devs"]["failures"][num_round] == 1)[0]

    def load_local_data(self, phase: str, dev_index: int):
        if phase == "fit":
            x = self.x_train[self.status.con["devs"]["local_data"][0][dev_index]]
            y = self.y_train[self.status.con["devs"]["local_data"][0][dev_index]]
        else:  # eval
            x = self.x_test[self.status.con["devs"]["local_data"][1][dev_index]]
            y = self.y_test[self.status.con["devs"]["local_data"][1][dev_index]]
        return x, y

