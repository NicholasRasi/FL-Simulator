import tensorflow as tf
from ..model_loader import DatasetModelLoader


class FashionMnist(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        return x_train, y_train, x_test, y_test

    # Image classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):  # https://www.tensorflow.org/tutorials/keras/classification
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        tf_model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=[metric])
        return tf_model

    def get_loss_function(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
