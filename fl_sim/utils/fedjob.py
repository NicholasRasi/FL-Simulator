from fl_sim.utils.fedphase import FedPhase


class FedJob:

    def __init__(self,
                 job_type,
                 num_round,
                 dev_index,
                 data,
                 config,
                 model_weights,
                 custom_loss):
        self.job_type = job_type
        self.num_round = num_round
        self.dev_index = dev_index
        self.data = data
        self.config = config
        self.model_weights = model_weights
        self.custom_loss = custom_loss

        self.num_examples = 0
        if job_type == FedPhase.FIT:
            self.mean_metric = None
            self.mean_loss = None
        elif job_type == FedPhase.EVAL:
            self.acc = None
            self.loss = None

    def __str__(self):
        return f"FedJob type: {self.job_type}, num_round: {self.num_round}, " \
               f"dev_index: {self.dev_index}, config: {self.config}"