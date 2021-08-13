import tensorflow as tf
import numpy as np
from keras.layers import BatchNormalization


class DatasetModelLoader:

    def __init__(self, model: str):
        self.model = model

    def get_dataset(self, mislabelling_percentage=0):
        x_train, y_train, x_test, y_test = self._get_dataset()

        if mislabelling_percentage > 0:
            # mislabelling: shuffle a percentage of labels
            # shuffle size
            print(y_train.shape[0])
            shuffle_size = int(mislabelling_percentage * y_train.shape[0])
            # take the indexes to shuffle
            shuffle_indexes = np.random.choice(y_train.shape[0], size=shuffle_size, replace=False)
            # take data at indexes
            shuffled = y_train[shuffle_indexes]
            # shuffle data
            np.random.shuffle(shuffled)
            # replace
            y_train[shuffle_indexes] = shuffled

        return x_train, y_train, x_test, y_test

    def _get_dataset(self):
        if self.model == "mnist":
            # load MNIST data
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0

            return x_train, y_train, x_test, y_test
        elif self.model == "fashion_mnist":  # https://www.tensorflow.org/tutorials/keras/classification
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            train_images = train_images / 255.0
            test_images = test_images / 255.0

            return train_images, train_labels, test_images, test_labels
        elif self.model == "cifar10":  # https://www.tensorflow.org/tutorials/images/cnn
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_images, test_images = train_images / 255.0, test_images / 255.0

            return train_images, train_labels, test_images, test_labels
        elif self.model == "boston_housing":  # https://keras.io/api/datasets/boston_housing
            (training_data, training_targets), (testing_data, testing_targets) = tf.keras.datasets.boston_housing.load_data()
            mean = training_data.mean(axis=0)
            training_data -= mean
            std = training_data.std(axis=0)
            training_data /= std

            testing_data -= mean
            testing_data /= std

            return training_data, training_targets, testing_data, testing_targets
        elif self.model == "imdb_reviews":  # https://builtin.com/data-science/how-build-neural-network-keras
            (training_data, training_targets), (testing_data, testing_targets) = tf.keras.datasets.imdb.load_data(num_words=10000)

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

    def get_compiled_model(self, optimizer: str, metric: str):
        if self.model == "mnist":  # https://www.tensorflow.org/tutorials/quickstart/beginner
            # build and compile Keras model
            tf_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            tf_model.compile(
                optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[metric]
            )
            return tf_model
        elif self.model == "fashion_mnist":  # https://www.tensorflow.org/tutorials/keras/classification
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10)
            ])
            tf_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=[metric])
            return tf_model
        elif self.model == "cifar10":  # https://www.tensorflow.org/tutorials/images/cnn
            tf_model = tf.keras.models.Sequential()
            tf_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
            tf_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            tf_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            tf_model.add(tf.keras.layers.Flatten())
            tf_model.add(tf.keras.layers.Dense(64, activation='relu'))
            tf_model.add(tf.keras.layers.Dense(10))

            tf_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=[metric])
            return tf_model
        elif self.model == "boston_housing":  # https://www.tensorflow.org/tutorials/keras/regression
            tf_model = tf.keras.models.Sequential()
            tf_model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)))
            tf_model.add(tf.keras.layers.Dense(64, activation='relu'))
            tf_model.add(tf.keras.layers.Dense(1))

            tf_model.compile(optimizer=optimizer, loss='mse', metrics=[metric])

            return tf_model
        elif self.model == "imdb_reviews":  # https://builtin.com/data-science/how-build-neural-network-keras
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
        else:
            return None


    @staticmethod
    def select_random_samples(y, num_clients, nk):
        clients_data_indexes = []
        for client in range(num_clients):
            indices = np.random.choice(y.shape[0], nk[client], replace=False)
            clients_data_indexes.append(indices)
        return clients_data_indexes

    @staticmethod
    def get_num_classes(y):
        return np.unique(y).shape[0]

    @staticmethod
    def get_classes(y):
        return np.unique(y)

    @staticmethod
    def select_non_iid_samples(y, num_clients, nk, alpha):
        # compute unique labels
        classes = DatasetModelLoader.get_num_classes(y)

        # create non-iid partitions
        data = DatasetModelLoader.non_iid_partition_with_dirichlet_distribution(label_list=y,
                                                                                client_num=num_clients,
                                                                                classes=classes,
                                                                                alpha=alpha)

        # get nk samples from partition
        clients_data_indexes = []
        for client in range(num_clients):
            client_indexes = data[client][0:nk[client]]
            clients_data_indexes.append(client_indexes)

        return clients_data_indexes

    @staticmethod
    def non_iid_partition_with_dirichlet_distribution(label_list,
                                                      client_num,
                                                      classes,
                                                      alpha,
                                                      task='classification'):
        """
            Obtain sample index list for each client from the Dirichlet distribution.
            This LDA method is first proposed by :
            Measuring the Effects of Non-Identical Data Distribution for
            Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).
            This can generate nonIIDness with unbalance sample number in each label.
            The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
            Dirichlet can support the probabilities of a K-way categorical event.
            In FL, we can view K clients' sample number obeys the Dirichlet distribution.
            For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution
            Parameters
            ----------
                label_list : the label list from classification/segmentation dataset
                client_num : number of clients
                classes: the number of classification (e.g., 10 for CIFAR-10) OR a list of segmentation categories
                alpha: a concentration parameter controlling the identicalness among clients.
                task: CV specific task eg. classification, segmentation
            Returns
            -------
                samples : ndarray,
                    The drawn samples, of shape ``(size, k)``.
        """
        net_dataidx_map = {}
        K = classes

        # For multiclass labels, the list is ragged and not a numpy array
        N = len(label_list) if task == 'segmentation' else label_list.shape[0]

        # guarantee the minimum number of sample in each client
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(client_num)]

            if task == 'segmentation':
                # Unlike classification tasks, here, one instance may have multiple categories/classes
                for c, cat in enumerate(classes):
                    if c > 0:
                        idx_k = np.asarray([np.any(label_list[i] == cat) and not np.any(
                            np.in1d(label_list[i], classes[:c])) for i in
                                            range(len(label_list))])
                    else:
                        idx_k = np.asarray(
                            [np.any(label_list[i] == cat) for i in range(len(label_list))])

                    # Get the indices of images that have category = c
                    idx_k = np.where(idx_k)[0]
                    idx_batch, min_size = DatasetModelLoader.partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k)
            else:
                # for each classification in the dataset
                for k in range(K):
                    # get a list of batch indexes which are belong to label k
                    idx_k = np.where(label_list == k)[0]
                    idx_batch, min_size = DatasetModelLoader.partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k)
        for i in range(client_num):
            np.random.shuffle(idx_batch[i])
            net_dataidx_map[i] = idx_batch[i]

        return net_dataidx_map

    @staticmethod
    def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
        np.random.shuffle(idx_k)
        # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
        # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))

        # get the index in idx_k according to the dirichlet distribution
        proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # generate the batch list for each client
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

        return idx_batch, min_size

    @staticmethod
    def record_data_stats(y_train, local_data_indexes):
        classes = DatasetModelLoader.get_classes(y_train)
        stats = []
        for indexes in local_data_indexes:
            count = []
            for i, cl in enumerate(classes):
                count.append(np.where(y_train[indexes] == cl)[0].shape[0])
            stats.append(count)
        return stats
