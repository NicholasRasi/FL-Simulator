import random
import tensorflow as tf
from ..model_loader import DatasetModelLoader
import numpy as np


class Mnist(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, y_train, x_test, y_test

    # Image classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data): # https://www.tensorflow.org/tutorials/quickstart/beginner
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

    def get_loss_function(self):
        return "sparse_categorical_crossentropy"

    def select_non_iid_samples(self, y, num_clients, nk, alpha):
        # compute unique labels
        classes = DatasetModelLoader.get_num_classes(y)

        # create non-iid partitions
        data = Mnist.non_iid_partition_2_ciphers_per_client(label_list=y,
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
    def non_iid_partition_2_ciphers_per_client(label_list,
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

        min_size_cat = 0
        while min_size_cat == 0:
            num_classes_per_client = 2
            classes_per_client = []

            for client in range(client_num):
                classes_per_client.append(random.sample(range(0, classes), 2))
            clients_per_classes = []

            for category in range(K):
                clients_per_class = []
                for i in range(client_num):
                    if category in classes_per_client[i]:
                        clients_per_class.append(i)
                clients_per_classes.append(clients_per_class)
            min_size_cat = min([len(x) for x in clients_per_classes])
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
                    idx_batch, min_size = Mnist.partition_class_samples_2_ciphers_per_client(N, alpha, client_num , idx_batch, idx_k, clients_per_classes[k])
            else:
                # for each classification in the dataset
                for k in range(K):
                    # get a list of batch indexes which are belong to label k
                    idx_k = np.where(label_list == k)[0]
                    idx_batch, min_size = Mnist.partition_class_samples_2_ciphers_per_client(N, alpha, client_num, idx_batch, idx_k, clients_per_classes[k])
        for i in range(client_num):
            np.random.shuffle(idx_batch[i])
            net_dataidx_map[i] = idx_batch[i]

        return net_dataidx_map

    @staticmethod
    def partition_class_samples_2_ciphers_per_client(N, alpha, num_tot_clients, idx_batch, idx_k, clients_per_class):
        proportions = [0]*num_tot_clients
        client_num = len(clients_per_class)
        np.random.shuffle(idx_k)
        # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
        # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
        prop = 1 / client_num
        for i in range(len(clients_per_class)):
          proportions[clients_per_class[i]] = prop

        # get the index in idx_k according to the dirichlet distribution
        proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])

        proportions = proportions / proportions.sum()

        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # generate the batch list for each client
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        return idx_batch, min_size
