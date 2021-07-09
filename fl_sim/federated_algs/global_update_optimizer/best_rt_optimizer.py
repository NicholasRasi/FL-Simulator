import numpy as np
from .global_update_optimizer import GlobalUpdateOptimizer


class BestRTOptimizer(GlobalUpdateOptimizer):

    def optimize(self, r: int, dev_index: int, phase: str) -> dict:
        # get the selected devices indexes
        selected_dev_indexes = np.where(self.status.var[phase]["devs"]["selected"][r] == 1)[0]
        # sort devices based on IPS
        sorted_devs_ips = np.argsort(self.status.con["devs"]["ips"])[::-1]
        # select the selected devices with max IPS
        max_ips_dev = -1
        for dev in sorted_devs_ips:
            if dev in selected_dev_indexes:
                max_ips_dev = dev
                break

        # compute the estimated round time (only computation) for the fastest (higher IPS) device
        iters = self.epochs * self.num_examples / self.batch_size
        est_round_time_max_ips = iters / self.status.con["devs"]["ips"][max_ips_dev]
        # scale the number of epochs
        est_round_time_cur_dev = iters / self.status.con["devs"]["ips"][dev_index]
        scale_epochs_factor = est_round_time_max_ips / est_round_time_cur_dev
        e = max(1, int(scale_epochs_factor * self.epochs))

        # self.logger.info("{} - {:.2f} - {:.2f}".format(dev_index, e*self.num_examples/self.batch_size / self.status.con["devs"]["ips"][dev_index], scale_epochs_factor))

        return {"epochs": e, "batch_size": self.batch_size, "num_examples": self.num_examples}
