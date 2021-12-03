import tensorflow as tf
import numpy as np
import tensorflow_datasets.public_api as tfds
from ..model_loader import DatasetModelLoader


class Emnist(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.tensorflow.org/datasets/catalog/emnist
        ds_train = tfds.load('emnist', split='train[:10%]')
        ds_test = tfds.load('emnist', split='test[:50%]')

        X_train = []
        y_train = []
        for sample in ds_train:
            X_train.append(sample['image'].numpy() / 255.0)
            y_train.append(sample['label'].numpy())

        X_test = []
        y_test = []
        for sample in ds_test:
            X_test.append(sample['image'].numpy() / 255.0)
            y_test.append(sample['label'].numpy())

        x_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        return x_train, y_train, x_test, y_test

    # Image classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):  # https://www.tensorflow.org/tutorials/quickstart/beginner
        tf_model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(62, activation="softmax"),
            ]
        )

        tf_model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[metric]
        )
        return tf_model

    def get_loss_function(self):
        return "sparse_categorical_crossentropy"
