import tensorflow as tf
from ..model_loader import DatasetModelLoader
import tensorflow_federated as tff
import numpy as np


class Shakespeare_tff(DatasetModelLoader):

    def to_ids(self, x):
        vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

        # Construct a lookup table to map string chars to indexes,
        # using the vocab loaded above:
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=vocab, values=tf.constant(list(range(len(vocab))),
                                               dtype=tf.int64)),
            default_value=0)

        s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids

    def split_input_target(self, chunk):
        input_text = tf.map_fn(lambda x: x[:-1], chunk)
        target_text = tf.map_fn(lambda x: x[1:], chunk)
        return (input_text, target_text)

    def preprocess(self, dataset):

        # Input pre-processing parameters
        SEQ_LENGTH = 100
        BATCH_SIZE = 1
        BUFFER_SIZE = 100  # For dataset shuffling

        return (
            # Map ASCII chars to int64 indexes using the vocab
            dataset.map(self.to_ids)
                # Split into individual chars
                .unbatch()
                # Form example sequences of SEQ_LENGTH +1
                .batch(SEQ_LENGTH + 1, drop_remainder=True)
                # Shuffle and form minibatches
                .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
                # And finally split into (input, target) tuples,
                # each of length SEQ_LENGTH.
                .map(self.split_input_target))

    @staticmethod
    def get_inputs_and_targets(federated_data):
        input_texts = []
        target_texts = []
        for x in federated_data:
            input_texts.append(x[0][0])
            target_texts.append(x[1][0])

        input_texts = np.array(input_texts)
        target_texts = np.array(target_texts)

        return input_texts, target_texts

    @staticmethod
    def get_indexes(federated_data):
        indexes_list = []
        i = 0
        for x in federated_data:
            client_indexes = []
            for sample in x:
                client_indexes.append(i)
                i = i + 1
            indexes_list.append(client_indexes)
        return indexes_list

    @staticmethod
    def join_samples(federated_data):
        all_elements = federated_data[0]

        for i in range(len(federated_data)):
            if i is not 0:
                all_elements = all_elements.concatenate(federated_data[i])
            i = i + 1
        return all_elements

    def get_dataset(self, mislabelling_percentage=0):
        # download dataset and load samples of clients
        train_data, test_data = tff.simulation.datasets.shakespeare.load_data(cache_dir="./datasets")
        sample_clients_train = train_data.client_ids[0:self.num_devices]
        sample_clients_test = test_data.client_ids[0:self.num_devices]
        federated_train_data = [(train_data.create_tf_dataset_for_client(x)) for x in sample_clients_train]
        federated_test_data = [(test_data.create_tf_dataset_for_client(x)) for x in sample_clients_test]

        # join clients samples
        all_elements_train = self.join_samples(federated_train_data)
        all_elements_test = self.join_samples(federated_test_data)

        # preprocess data samples
        example_dataset_train = self.preprocess(all_elements_train)
        example_dataset_test = self.preprocess(all_elements_test)

        # split input from targets
        input_texts_train, target_texts_train = self.get_inputs_and_targets(example_dataset_train)
        input_texts_test, target_texts_test = self.get_inputs_and_targets(example_dataset_test)

        return input_texts_train, target_texts_train, input_texts_test, target_texts_test

    # Text generation task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):  # https://www.tensorflow.org/text/tutorials/text_generation

        class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

            def __init__(self, name='accuracy', dtype=tf.float32):
                super().__init__(name, dtype=dtype)

            def update_state(self, y_true, y_pred, sample_weight=None):
                vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
                y_true = tf.reshape(y_true, [-1, 1])
                y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
                return super().update_state(y_true, y_pred, sample_weight)

        x_train, y_train = train_data

        # Length of the vocabulary in chars
        vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
        vocab_size = len(vocab)

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

        idx2char = np.array(vocab)
        tf_model = MyModel(
            # Be sure the vocabulary size matches the `StringLookup` layers.
            vocab_size=len(idx2char),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

        tf_model.build(input_shape=x_train.shape)

        tf_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[FlattenedCategoricalAccuracy()])

        return tf_model

    def get_loss_function(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def select_non_iid_samples(self, y, num_clients, nk, alpha):
        # download dataset and load samples of clients
        train_data, test_data = tff.simulation.datasets.shakespeare.load_data()
        sample_clients_train = train_data.client_ids[0:self.num_devices]
        federated_train_data = [(train_data.create_tf_dataset_for_client(x)) for x in sample_clients_train]

        return self.get_indexes(federated_train_data)



