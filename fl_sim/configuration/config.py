import yaml


class Config:

    def __init__(self, config_file):
        # read config file
        with open(config_file, 'r') as f:
            config = yaml.load(f)

        self.simulation = config["simulation"]
        self.algorithms = config["algorithms"]
        self.devices = config["devices"]
        self.computation = config["computation"]
        self.energy = config["energy"]
        self.network = config["network"]
        self.data = config["data"]
