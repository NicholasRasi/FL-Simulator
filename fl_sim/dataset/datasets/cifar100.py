import tensorflow as tf
from ..model_loader import DatasetModelLoader


class Cifar100(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test

    # Image classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):  # https://www.tensorflow.org/tutorials/images/cnn
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

    def get_loss_function(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
