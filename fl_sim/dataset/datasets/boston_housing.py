import tensorflow as tf
from ..model_loader import DatasetModelLoader


class BostonHousing(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing
        (training_data, training_targets), (testing_data, testing_targets) = tf.keras.datasets.boston_housing.load_data()
        mean = training_data.mean(axis=0)
        training_data -= mean
        std = training_data.std(axis=0)
        training_data /= std

        testing_data -= mean
        testing_data /= std
        return training_data, training_targets, testing_data, testing_targets

    # Regression task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):
        tf_model = tf.keras.models.Sequential()
        tf_model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)))
        tf_model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf_model.add(tf.keras.layers.Dense(1))

        tf_model.compile(optimizer=optimizer, loss='mse', metrics=[metric])

        return tf_model

    def get_loss_function(self):
        return "mse"