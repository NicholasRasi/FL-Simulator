import tensorflow as tf
import urllib
from urllib.request import urlopen
import numpy as np
import pandas as pd
import tarfile
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from ..model_loader import DatasetModelLoader


class Wisdm(DatasetModelLoader):

    def get_dataset(self, mislabelling_percentage=0):  # https://www.cis.fordham.edu/wisdm/dataset.php
        url = 'https://www.cis.fordham.edu/wisdm/includes/datasets/latest/'
        dataset_url = 'WISDM_ar_latest.tar.gz'
        r = urlopen(url + dataset_url)
        t = tarfile.open(name=None, fileobj=BytesIO(r.read()), encoding='UTF-8')
        f = t.extractfile('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

        # Read lines from txt

        processedLines = []

        lines = f.readlines()

        for i, line in enumerate(lines):
            try:
                line = line.decode("utf-8").split(',')
                last = line[5].split(';')[0]
                last = last.strip()
                if last == '':
                    break
                temp = [line[0], line[1], line[2], line[3], line[4], last]
                processedLines.append(temp)
            except:
                pass

        # Build dataframe

        columns = ['user', 'activity', 'time', 'x', 'y', 'z']
        data = pd.DataFrame(data=processedLines, columns=columns)

        data['x'] = data['x'].astype('float')
        data['y'] = data['y'].astype('float')
        data['z'] = data['z'].astype('float')

        df = data.drop(['user', 'time'], axis=1).copy()

        label = LabelEncoder()
        df['label'] = label.fit_transform(df['activity'])

        X = df[['x', 'y', 'z']]
        y = df['label']

        # Standardize data

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        scaled_X = pd.DataFrame(data=X, columns=['x', 'y', 'z'])
        scaled_X['label'] = y.values

        # Frame preparation

        Fs = 20
        frame_size = Fs * 4
        hop_size = Fs * 2

        X, y = self.get_frames(scaled_X, frame_size, hop_size)

        # Split train and test sets

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        #  Reshape for CNN

        X_train = X_train.reshape(X_train.shape[0], 80, 3, 1)
        X_test = X_test.reshape(X_test.shape[0], 80, 3, 1)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def get_frames(df, frame_size, hop_size):
        N_FEATURES = 3

        frames = []
        labels = []
        for i in range(0, len(df) - frame_size, hop_size):
            x = df['x'].values[i: i + frame_size]
            y = df['y'].values[i: i + frame_size]
            z = df['z'].values[i: i + frame_size]

            label = stats.mode(df['label'][i: i + frame_size])[0][0]
            frames.append([x, y, z])
            labels.append(label)

        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        labels = np.asarray(labels)

        return frames, labels

    # Activity recognition task
    def get_compiled_model(self, optimizer: str, metric: str, train_data):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, (2, 2), activation="relu", input_shape=(80, 3, 1)))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Conv2D(32, (2, 2), activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(6, activation='softmax'))

        model.compile(optimizer=optimizer, loss='mse', metrics=[metric])

        return model

    def get_loss_function(self):
        return tf.keras.losses.SparseCategoricalCrossentropy()
