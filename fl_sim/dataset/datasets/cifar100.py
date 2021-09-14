import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from fl_sim.configuration import Config
from ..model_loader import DatasetModelLoader


class Cifar100(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test

    def get_compiled_model(self, optimizer: str, metric: str):
        tf_model = tf.keras.models.Sequential()
        tf_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        tf_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        tf_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        tf_model.add(tf.keras.layers.Flatten())
        tf_model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf_model.add(tf.keras.layers.Dense(100))

        tf_model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=[metric])
        return tf_model