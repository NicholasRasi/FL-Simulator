import tensorflow as tf
from sklearn.model_selection import train_test_split
from ..model_loader import DatasetModelLoader
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing


class Shakespeare(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):
        path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

        # VECTORIZE

        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

        # The unique characters in the file
        vocab = sorted(set(text))

        # Define preprocessing function to convert chars into ids
        ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(vocab), mask_token=None)

        # CREATE TRAINING EXAMPLES AND TARGETS

        # Apply preprocessing to obtain ids
        all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        # Define the length of each sequence
        seq_length = 100

        # Divide into batches
        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

        # Define the function which builds up input-target pairs
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        # Obtain the dataset composed of pairs input-target
        dataset = sequences.map(split_input_target)

        # CREATE TRAINING BATCHES

        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        BUFFER_SIZE = 10000

        dataset = (
            dataset
                .shuffle(BUFFER_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))

        x_train = []
        y_train = []

        for elem in dataset:
            x_train.append(elem[0])
            y_train.append(elem[1])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

        return x_train, y_train, x_test, y_test

    def get_compiled_model(self, optimizer: str, metric: str, train_data):  # https://www.tensorflow.org/text/tutorials/text_generation

        x_train, y_train = train_data

        # Length of the vocabulary in chars
        vocab_size = 65

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        class MyModel(tf.keras.Model):
            def __init__(self, vocab_size, embedding_dim, rnn_units):
                super().__init__(self)
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                self.gru = tf.keras.layers.GRU(rnn_units,
                                               return_sequences=True,
                                               return_state=True)
                self.dense = tf.keras.layers.Dense(vocab_size)

            def call(self, inputs, states=None, return_state=False, training=False):
                x = inputs
                x = self.embedding(x, training=training)
                if states is None:
                    states = self.gru.get_initial_state(x)
                x, states = self.gru(x, initial_state=states, training=training)
                x = self.dense(x, training=training)

                if return_state:
                    return x, states
                else:
                    return x

        tf_model = MyModel(
            # Be sure the vocabulary size matches the `StringLookup` layers.
            vocab_size=66,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        tf_model.build(input_shape=x_train.shape)

        tf_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        return tf_model

    def get_loss_function(self):
        return tf.losses.SparseCategoricalCrossentropy(from_logits=True)

