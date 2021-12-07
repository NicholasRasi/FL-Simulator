import re
import string
import numpy as np
import tensorflow as tf
from ..model_loader import DatasetModelLoader
import tensorflow_datasets as tfds


class Shakespeare(DatasetModelLoader):

    def __init__(self, num_devices: int):
        super().__init__(num_devices)
        self.vocabulary = None

    def get_dataset(self, mislabelling_percentage=0):
        ds = tfds.load('tiny_shakespeare', split='train')
        text = None

        for x in ds:
            text = x['text'].numpy()

        # Clean up text
        text = text.decode(encoding='utf-8')
        text = re.sub(r'.+:\n\b', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.replace('\n', ' ')
        text = text.lower()
        text = text[:160000]

        self.vocabulary = sorted(list(set(text)))

        # Select subset of dataset since it's huge (1.600.000 examples)
        text_train = text[:100000]
        text_test = text[100000:160000]

        x_enc_train, y_enc_train = self.get_encoded_dataset(text_train)
        x_enc_test, y_enc_test = self.get_encoded_dataset(text_test)

        return x_enc_train.numpy(), y_enc_train.numpy(), x_enc_test.numpy(), y_enc_test.numpy()

    def get_encoded_dataset(self, text):

        # Dictionaries for char-to-int/int-to-char conversion
        ctoi = {c: i for i, c in enumerate(self.vocabulary)}

        seq_length = 100
        full_text_length = len(text)

        X_enc = []
        Y_enc = []
        # Cycle over the full text
        step = 1
        for i in range(0, full_text_length - (seq_length), step):
            sequence = text[i:i + seq_length]
            target = text[i + 1:i + seq_length + 1]
            X_enc.append([ctoi[c] for c in sequence])
            Y_enc.append([ctoi[c] for c in target])

        X_enc = tf.one_hot(np.array(X_enc), len(self.vocabulary))
        Y_enc = tf.one_hot(np.array(Y_enc), len(self.vocabulary))

        return X_enc, Y_enc

    # Text generation task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):

        # Hidden size (state)
        h_size = 128

        # Model
        input_x = tf.keras.Input(shape=(None, len(self.vocabulary)))

        lstm1 = tf.keras.layers.LSTM(
            units=h_size, batch_input_shape=[None, None, len(self.vocabulary)],
            return_sequences=True, return_state=True, stateful=False)
        lstm2 = tf.keras.layers.LSTM(
            units=h_size, return_sequences=True,
            return_state=True, stateful=False)
        dense = tf.keras.layers.Dense(units=len(self.vocabulary), activation='softmax')

        x, _, _ = lstm1(input_x)
        x, _, _ = lstm2(x)
        out = dense(x)

        train_model = tf.keras.Model(
            inputs=input_x, outputs=out)

        train_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[metric])

        return train_model

    def get_loss_function(self):
        return tf.losses.CategoricalCrossentropy()