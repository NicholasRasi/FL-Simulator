import tensorflow as tf
import tensorflow_datasets as tfds
from ..model_loader import DatasetModelLoader
import numpy as np


class OxfordPets(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet
        dataset = tfds.load('oxford_iiit_pet:3.*.*')
        train_images = dataset['train'].map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_images = dataset['test'].map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)

        BUFFER_SIZE = 1000
        train_batches = (
            train_images
                .cache()
                .shuffle(BUFFER_SIZE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

        test_batches = test_images

        x_train = []
        y_train = []
        for image in train_batches:
            x_train.append(image[0].numpy())
            y_train.append(image[1].numpy())

        x_test = []
        y_test = []
        for image in test_batches:
            x_test.append(image[0].numpy())
            y_test.append(image[1].numpy())

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        return x_train, y_train, x_test, y_test

    def load_image(self, datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask

    def normalize(self, input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    # Image segmentation task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):

        depth = 2
        num_classes = 21

        model = tf.keras.Sequential()

        # Encoder
        # -------
        img_h = 128
        img_w = 128

        vgg = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(img_h, img_w, 3))
        for layer in vgg.layers:
            layer.trainable = False

        model.add(vgg)

        start_f = 256

        # Decoder
        # -------
        for i in range(depth):
            model.add(tf.keras.layers.UpSampling2D(2, interpolation='bilinear'))
            model.add(tf.keras.layers.Conv2D(filters=start_f,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding='same'))
            model.add(tf.keras.layers.ReLU())

            start_f = start_f // 2

        # Prediction Layer
        # ----------------
        model.add(tf.keras.layers.Conv2D(filters=num_classes,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         activation='softmax'))

        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[metric]
        )
        model.summary()
        return model

    def get_loss_function(self):
        return "sparse_categorical_crossentropy"
