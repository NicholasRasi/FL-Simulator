import tensorflow as tf
import tensorflow_federated as tff
from ..model_loader import DatasetModelLoader
import numpy as np


class Emnist_tff(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):
        (emnist_train), (emnist_test) = tff.simulation.datasets.emnist.load_data(cache_dir="./datasets")

        sample_clients_train = emnist_train.client_ids[0:self.num_devices]
        sample_clients_test = emnist_test.client_ids[0:self.num_devices]

        federated_train_data = [(emnist_train.create_tf_dataset_for_client(x)) for x in sample_clients_train]
        federated_test_data = [(emnist_test.create_tf_dataset_for_client(x)) for x in sample_clients_test]

        images_train, labels_train = self.get_images_and_labels(federated_train_data)
        images_test, labels_test = self.get_images_and_labels(federated_test_data)

        return images_train, labels_train, images_test, labels_test

    def get_compiled_model(self, optimizer: str, metric: str):  # https://www.tensorflow.org/tutorials/quickstart/beginner
        # build and compile Keras model

        tf_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(10, kernel_initializer='zeros'),
                tf.keras.layers.Softmax(),
            ]
        )

        tf_model.compile(
            optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[metric]
        )
        return tf_model

    @staticmethod
    def get_images_and_labels(federated_data):
        images_numpy = []
        labels_numpy = []

        for client_data in federated_data:
            for client_image in client_data:
                images_numpy.append(np.float64(client_image['pixels'].numpy()))
                labels_numpy.append(client_image['label'].numpy())

        return np.array(images_numpy), np.array(labels_numpy)

    @staticmethod
    def get_images_and_labels_by_client(federated_data):
        images_numpy = []
        labels_numpy = []

        for client_data in federated_data:
            images_of_client = []
            labels_of_client = []
            for client_image in client_data:
                images_of_client.append(np.float64(client_image['pixels'].numpy()))
                labels_of_client.append(client_image['label'].numpy())
            images_numpy.append(images_of_client.numpy())
            labels_numpy.append(labels_of_client.numpy())

        return np.array(images_numpy), np.array(labels_numpy)

    def select_non_iid_samples(self, y, num_clients, nk, alpha):

        (emnist_train), (emnist_test) = tff.simulation.datasets.emnist.load_data()

        sample_clients_train = emnist_train.client_ids[0:num_clients]

        federated_train_data = [emnist_train.create_tf_dataset_for_client(x) for x in sample_clients_train]

        images_train, labels_train = self.get_images_and_labels_by_client(federated_train_data)

        clients_data_indexes = []
        i = 0

        for client_images in images_train:
            client_indexes = []
            for image in client_images:
                client_indexes.append(i)
                i = i + 1
            clients_data_indexes.append(client_indexes)

        return clients_data_indexes
