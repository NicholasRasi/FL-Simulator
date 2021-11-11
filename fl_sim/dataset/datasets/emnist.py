import tensorflow as tf
import numpy as np
import tensorflow_datasets.public_api as tfds
from sklearn.model_selection import train_test_split
from ..model_loader import DatasetModelLoader


class Emnist(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.tensorflow.org/datasets/catalog/emnist
        ds = tfds.load('emnist', split='train[:10%]')

        X = []
        y = []

        for sample in ds:
            X.append(sample['image'].numpy() / 255.0)
            y.append(sample['label'].numpy())

        X = np.asarray(X)
        y = np.asarray(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        return x_train, y_train, x_test, y_test

    # Image classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=1))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(62, activation='softmax'))

        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[metric]
        )
        return model

    def get_loss_function(self):
        return "sparse_categorical_crossentropy"
