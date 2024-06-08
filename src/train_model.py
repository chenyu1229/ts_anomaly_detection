import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Model

def load_dataset(path):
    """
    load dataset from local or external url
    :param path: the location of dataset
    :return: dataset in Dataframe format
    """
    df = pd.read_csv(path, header=None)
    return df

def preprocessing(df):
    """
    preprcessing including drop missing value, normlization, and split dataset into training set and testing set
    :param df: input dataset
    :return: normal ECG training set, abnormal ECG training set, testing set, training label, testing label
    """
    df=df.dropna()
    data = df.iloc[:,:-1].values
    labels = df.iloc[:,-1].values
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 86)
    #Now lets Normalize the data
    #First we will calculate the maximum and minimum value from the training set 
    mean = tf.math.reduce_mean(train_data)
    std = tf.math.reduce_std(train_data)
    
    #Now we will use the formula (data - min)/(max - min)
    train_data = (train_data - mean)/std
    test_data = (test_data - mean)/std
    
    #I have converted the data into float
    train_data = tf.cast(train_data, dtype=tf.float32)
    test_data = tf.cast(test_data, dtype=tf.float32)
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)
    #Now let's separate the data for normal ECG from that of abnormal ones
    #Normal ECG data
    n_train_data = train_data[train_labels]
    # n_test_data = test_data[test_labels]
    
    #Abnormal ECG data
    an_train_data = train_data[~train_labels]
    # an_test_data = test_data[~test_labels]
    x_n_train = np.expand_dims(n_train_data,axis=2)
    x_an_train = np.expand_dims(an_train_data,axis=2)
    x_test = np.expand_dims(test_data,axis=2)
    return x_n_train, x_an_train, x_test, train_labels, test_labels

def build_autoencoder(x_train):
    n_steps = x_train.shape[1]
    n_features = x_train.shape[2]
    model = keras.Sequential(
        [
            layers.Input(shape=(n_steps, n_features)),
            layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',
                dilation_rate=1, activation="linear"),
            layers.LSTM(
                units=25, activation="tanh", name="lstm_1", return_sequences=False
            ),
            layers.RepeatVector(n_steps),
            layers.LSTM(
                units=25, activation="tanh", name="lstm_2", return_sequences=True
            ),
            layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',
                dilation_rate=1, activation="linear"),
            layers.TimeDistributed(layers.Dense(x_train.shape[2], activation='linear'))
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    return model

def save_mae_graph(train_mae_loss,train_mae_loss_an,path):
    plt.subplots(figsize=(12,6))
    plt.hist(train_mae_loss, bins=50,label="normal", alpha=.6, color="blue")
    plt.hist(train_mae_loss_an, bins=50,label="abnormal", alpha=.6, color="red")
    plt.xlabel("Training MAE")
    plt.ylabel("No of samples")
    plt.legend()
    plt.xticks(np.arange(0, 1.6, 0.1)) 
    plt.savefig(path)


def print_stats(predictions, labels):
    print("Accuracy = {:.2%}".format(accuracy_score(labels, predictions)))
    print("Precision = {:.2%}".format(precision_score(labels, predictions)))
    print("Recall = {:.2%}".format(recall_score(labels, predictions)))

df = load_dataset('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv')
x_n_train, x_an_train, x_test, train_labels, test_labels = preprocessing(df)
autoencoder  = build_autoencoder(x_n_train)
history = autoencoder.fit(x_n_train,
                          x_n_train,
                          epochs=150,
                          batch_size=128,
                          validation_split=0.1,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")
                        ],
)
autoencoder.save('../models/autoencoder_model.keras') 
x_n_train_pred = autoencoder.predict(x_n_train)
x_an_train_pred = autoencoder.predict(x_an_train)
n_train_mae_loss = np.mean(np.abs(x_n_train_pred - x_n_train), axis=1)
an_train_mae_loss = np.mean(np.abs(x_an_train_pred - x_an_train), axis=1)
save_mae_graph(n_train_mae_loss,an_train_mae_loss,"../reports/training_MAE_loss.jpg")

print("Predicting on testing dataset")
x_test_pred = autoencoder.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
anomalies = test_mae_loss < 0.3

print("Calculating accuracy on testing dataset")
print_stats(anomalies,test_labels)

