from abc import abstractmethod, ABC
from typing import List
from fl_sim.configuration import Config
from fl_sim import Status
import numpy as np


class ClientsSelector(ABC):

    def __init__(self, config: Config, status: Status, logger):
        self.config = config
        self.status = status
        self.logger = logger

    @abstractmethod
    def select_devices(self, num_round: int) -> List:
        """
        Select devices
        :param num_round: the devices index selected for the current round
        """

    def get_available_devices(self, num_round):
        avail_devs = self.status.con["devs"]["availability"][num_round]
        avail_indexes = np.where(avail_devs == 1)[0]
        return avail_indexes
