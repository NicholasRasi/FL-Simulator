import tensorflow as tf
from ..model_loader import DatasetModelLoader
import numpy as np


class Mnist(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_images_and_labels(federated_data):
        images_numpy = []
        labels_numpy = []

        for client_data in federated_data:
            for client_image in client_data:
                images_numpy.append(client_image['pixels'].numpy())
                labels_numpy.append(client_image['label'].numpy())

        return np.array(images_numpy), np.array(labels_numpy)

    def get_compiled_model(self, optimizer: str, metric: str, train_data): # https://www.tensorflow.org/tutorials/quickstart/beginner
        # build and compile Keras model
        tf_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        tf_model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[metric]
        )
        return tf_model