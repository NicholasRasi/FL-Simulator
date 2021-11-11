from .datasets.mnist import Mnist
from .datasets.emnist import Emnist
from .datasets.fashion_mnist import FashionMnist
from .datasets.cifar10 import Cifar10
from .datasets.cifar100 import Cifar100
from .datasets.oxford_pets import Oxford_pets
from .datasets.tff_cifar100 import Cifar100_tff
from .datasets.tff_emnist import Emnist_tff
from .datasets.boston_housing import BostonHousing
from .datasets.imdb_reviews import ImdbReviews
from .datasets.shakespeare import Shakespeare
from .datasets.tff_shakespeare import Shakespeare_tff
from .datasets.sentiment140 import Sentiment140


class DatasetModelLoaderFactory:

    @staticmethod
    def get_model_loader(model_name: str, num_devices: int):
        if model_name == "mnist":
            return Mnist(num_devices)
        if model_name == "emnist":
            return Emnist(num_devices)
        elif model_name == "tff_emnist":
            return Emnist_tff(num_devices)
        elif model_name == "fashion_mnist":
            return FashionMnist(num_devices)
        elif model_name == "cifar10":
            return Cifar10(num_devices)
        elif model_name == "cifar100":
            return Cifar100(num_devices)
        elif model_name == "tff_cifar100":
            return Cifar100_tff(num_devices)
        if model_name == "oxford_pets":
            return Oxford_pets(num_devices)
        elif model_name == "boston_housing":
            return BostonHousing(num_devices)
        elif model_name == "imdb_reviews":
            return ImdbReviews(num_devices)
        elif model_name == "shakespeare":  # https://www.tensorflow.org/text/tutorials/text_generation
            return Shakespeare(num_devices)
        elif model_name == "tff_shakespeare":  # https://www.tensorflow.org/text/tutorials/text_generation
            return Shakespeare_tff(num_devices)
        elif model_name == "sentiment140":  # https://www.tensorflow.org/text/tutorials/text_classification_rnn
            return Sentiment140(num_devices)
        else:
            return None
