import yaml


class Config:

    def __init__(self, config_file):
        self.simulation = None
        self.algorithms = None
        self.devices = None
        self.computation = None
        self.energy = None
        self.network = None
        self.data = None

        self.config = self.read_config(config_file)

        self.simulation = self.config["simulation"]
        self.algorithms = self.config["algorithms"]
        self.devices = self.config["devices"]
        self.computation = self.config["computation"]
        self.energy = self.config["energy"]
        self.network = self.config["network"]
        self.data = self.config["data"]

    def read_config(self, config_file):
        config = Config.read_file(config_file)

        if "import" in config:
            import_file = config.pop("import")
            return Config._merge_dictionaries(self.read_config(import_file), config)
        else:
            return config

    @staticmethod
    def read_file(config_file):
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config

    @staticmethod
    def _merge_dictionaries(dict1, dict2):
        """
        Recursive merge dictionaries.

        :param dict1: Base dictionary to merge.
        :param dict2: Dictionary to merge on top of base dictionary.
        :return: Merged dictionary
        """
        for key, val in dict1.items():
            if isinstance(val, dict):
                dict2_node = dict2.setdefault(key, {})
                Config._merge_dictionaries(val, dict2_node)
            else:
                if key not in dict2:
                    dict2[key] = val

        return dict2
