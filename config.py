import os


class Config:

    def __init__(self):
        # SIMULATION
        self.simulation_output_folder = str(os.getenv('FL_OUT_FOLDER') or "output")

        # file where to save the simulation output data
        self.simulation_output_file = str(os.getenv('FL_OUT_FILE') or "base.json")

        # model used for the simulation
        self.model_name = str(os.getenv('FL_MODEL') or "mnist")

        # total number of rounds
        self.num_rounds = int(os.getenv('FL_ROUNDS') or 10)

        # algorithm used for the FL aggregation
        self.aggregation = str(os.getenv('FL_ALG') or "fedavg")

        # algorithm used for the clients selection (fit)
        self.selection_fit = str(os.getenv('FL_SEL_F') or "random")

        # algorithm used for the update optimizer (fit)
        self.global_upd_opt_fit = str(os.getenv('FL_G_UPD_OPT_F') or "static")

        # algorithm used for the local data optimizer (fit)
        self.local_data_opt_fit = str(os.getenv('FL_L_DAT_OPT_E') or "random")

        # algorithm used for the clients selection (eval)
        self.selection_eval = str(os.getenv('FL_SEL_E') or "random")

        # algorithm used for the update optimizer (eval)
        self.global_upd_opt_eval = str(os.getenv('FL_G_UPD_OPT_E') or "static")

        # algorithm used for the local data optimizer (eval)
        self.local_data_opt_eval = str(os.getenv('FL_L_DAT_OPT_E') or "random")

        # use i.d.d data
        # self.iid = bool(os.getenv('FL_IID') or True)

        # optimizer used (local iteration) string or optimizer instance
        self.optimizer = str(os.getenv('FL_OPT') or "sgd")

        # fixing random state for reproducibility
        self.random_seed = 19680801

        # set the TensorFlow verbosity, 0 = silent, 1 = progress bar, 2 = one line per epoch
        self.tf_verbosity = 0

        # DEVICES
        # total number of devices (D)
        self.num_devs = int(os.getenv('FL_MIN') or 50)

        # probability a device is available for a round
        self.p_available = float(os.getenv('FL_P_AVAIL') or 1)

        # probability a device fails during a round
        self.p_fail = float(os.getenv('FL_P_FAIL') or 0)

        # LOCAL ITERATIONS
        # number of epochs executed for fit for each round
        self.epochs = int(os.getenv('FL_EPOCHS') or 10)

        # batch size used for fit for each round
        self.batch_size_fit = int(os.getenv('FL_BATCHSIZE_F') or 32)

        # number of examples used for fit for each round
        self.num_examples_fit = int(os.getenv('FL_NK_F') or 100)

        # batch size used for evaluation for each round
        self.batch_size_eval = int(os.getenv('FL_BATCHSIZE_E') or self.batch_size_fit)

        # number of examples used for evaluation for each round
        self.num_examples_eval = int(os.getenv('FL_NK_E') or self.num_examples_fit)

        # fraction of clients used for the computation (fit)
        self.k_fit = float(os.getenv('FL_K_FIT') or 0.5)

        # fraction of client used for the computation (fit)
        self.k_eval = float(os.getenv('FL_K_EVAL') or self.k_fit)

        # COMPUTATION
        # mean and variance of number of computed iterations/second per device (among devices) [iter/s]
        self.ips_mean = int(os.getenv('FL_IPS_MEAN') or 100)
        self.ips_var = int(os.getenv('FL_IPS_VAR') or 50)

        # ENERGY
        # mean and variance of energy available at each device [mWh]
        self.energy_mean = int(os.getenv('FL_ENE_MEAN') or 3000)
        self.energy_var = int(os.getenv('FL_ENE_VAR') or 1000)

        # power consumption for 1 second of computation [mW/s]
        self.pow_comp_s = int(os.getenv('FL_ENE_COM_S') or 100)
        # power consumption for 1 second of network used [mW/s]
        self.pow_net_s = int(os.getenv('FL_ENE_NET_S') or 200)

        # NETWORK
        # mean and variance of network speed available at each device [params/s]
        self.netspeed_mean = int(os.getenv('FL_NETSPEED_MEAN') or 1000000)
        self.netspeed_var = int(os.getenv('FL_NETSPEED_VAR') or 0)

        # LOCAL DATA
        # mean and variance of number of examples available at each device
        self.local_data_mean = int(os.getenv('FL_A_MEAN') or 100)
        self.local_data_var = int(os.getenv('FL_A_VAR') or 0)

        # increment of the local dataset size at each round
        # self.local_data_incr_round = int(os.getenv('FL_A_INCR_STEP') or 0)

        # TRANSMITTED DATA
        # fraction of model weights transmitted
        # self.frac_trans_weights = float(os.getenv('FL_V') or 1)
