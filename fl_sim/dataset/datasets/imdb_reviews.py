import tensorflow as tf
import numpy as np
from ..model_loader import DatasetModelLoader


class ImdbReviews(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):

        (training_data, training_targets), (testing_data, testing_targets) = tf.keras.datasets.imdb.load_data(
            num_words=10000)

        # prepare data
        def vectorize(sequences, dimension=10000):
            results = np.zeros((len(sequences), dimension))

            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1
            return results

        training_data = vectorize(training_data)
        testing_data = vectorize(testing_data)
        training_targets = np.array(training_targets).astype("float32")
        testing_targets = np.array(testing_targets).astype("float32")

        return training_data, training_targets, testing_data, testing_targets

    # Text classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):

        # Input - Layer
        tf_model = tf.keras.models.Sequential()
        tf_model.add(tf.keras.layers.Dense(50, activation="relu", input_shape=(10000,)))
        # Hidden - Layers
        tf_model.add(tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None))
        tf_model.add(tf.keras.layers.Dense(50, activation="relu"))
        tf_model.add(tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None))
        tf_model.add(tf.keras.layers.Dense(50, activation="relu"))
        # Output- Layer
        tf_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        tf_model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[metric]
        )

        return tf_model

    def get_loss_function(self):
        return "binary_crossentropy"

