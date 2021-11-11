import tensorflow as tf
import urllib
from urllib.request import urlopen
import numpy as np
import pandas as pd
import re
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from ..model_loader import DatasetModelLoader


class Sentiment140(DatasetModelLoader):

    def __init__(self, num_devices: int):
        super().__init__(num_devices)
        self.users = []
        self.encoder = None

    def get_dataset(self, mislabelling_percentage=0):  # http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

        # Load dataset
        url = 'http://cs.stanford.edu/people/alecmgo/'
        dataset_url = 'trainingandtestdata.zip'
        resp = urllib.request.urlopen(url + urllib.request.quote(dataset_url))
        zipfile = ZipFile(BytesIO(resp.read()))
        DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
        DATASET_ENCODING = "ISO-8859-1"
        data = TextIOWrapper(zipfile.open('training.1600000.processed.noemoticon.csv'), encoding=DATASET_ENCODING)
        df = pd.read_csv(data, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

        # Reshuffle
        df = df.sample(frac=1)

        # Removing the unnecessary columns.
        df = df[['sentiment', 'text', 'user']]

        # Replace 4 with 1
        df['sentiment'] = df['sentiment'].replace(4, 1)

        df = df[:30000]

        # Storing data in lists.
        text, sentiments = list(df['text']), list(df['sentiment'])

        # Preprocess text
        processedtext = self.preprocess(text)

        # Split data
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(processedtext, sentiments, train_size=0.8, test_size=0.2, shuffle=False)
        #self.users = list(df['user'])[:int(0.8*len(processedtext))]
        train_inputs = np.asarray(train_inputs)
        train_targets = np.asarray(train_targets)
        test_inputs = np.asarray(test_inputs)
        test_targets = np.asarray(test_targets)

        VOCAB_SIZE = 1000

        encoder = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE)
        encoder.adapt(train_inputs)
        vocab = np.array(encoder.get_vocabulary())

        self.encoder = encoder

        return train_inputs, train_targets, test_inputs, test_targets

    # Text classification task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):  # https://www.tensorflow.org/text/tutorials/text_classification_rnn

        model = tf.keras.Sequential([
            self.encoder,
            tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=64,
                # Use masking to handle the variable sequence lengths
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=optimizer,
                      metrics=[metric])

        return model

    def get_loss_function(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)


    # dictionary containing all emojis.
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    # set containing all stopwords.
    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                    'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                    'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                    'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                    's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                    'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                    'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                    'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                    "youve", 'your', 'yours', 'yourself', 'yourselves']

    def preprocess(self, textdata):

        processedtext = []

        # Defining regex patterns.
        urlpattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userpattern = '@[^\s]+'
        alphapattern = "[^a-zA-Z0-9]"
        sequencepattern = r"(.)\1\1+"
        seqreplacepattern = r"\1\1"

        for tweet in textdata:
            tweet = tweet.lower()

            # Replace all URls with 'URL'
            tweet = re.sub(urlpattern, ' URL', tweet)
            # Replace all emojis.
            for emoji in self.emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + self.emojis[emoji])
                # Replace @USERNAME to 'USER'.
            tweet = re.sub(userpattern, ' USER', tweet)
            # Replace all non alphabets.
            tweet = re.sub(alphapattern, " ", tweet)
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencepattern, seqreplacepattern, tweet)

            processedtext.append(tweet)

        return processedtext

    @staticmethod
    def get_numpy_array(dataset):
        x = []
        y = []
        for elem in dataset:
            x.append(elem[0].numpy())
            y.append(elem[1].numpy())

        x = np.array(x)
        y = np.array(y)

        return x, y

    # Each device has reviews belonging to a specific user
    def select_non_iid_samples(self, y, num_clients, nk, alpha):
        train_users_no_duplicates = list(set(self.users))
        clients_data_indexes = []
        for client in range(num_clients):
            client_indexes = [i for i in range(len(self.users)) if self.users[i] == train_users_no_duplicates[client]][0:nk[client]]
            clients_data_indexes.append(client_indexes)

        return clients_data_indexes

