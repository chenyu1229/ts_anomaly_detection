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
    :return:    normal ECG training set
                abnormal ECG training set
                testing set
                training label
                testing label
    """
    #delete missing value in dataset
    df=df.dropna()
    data = df.iloc[:,:-1].values
    labels = df.iloc[:,-1].values
    #split dataset into training set, testing set, training labels and test labels
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 86)
    #Normalize the data
    #Calculate the mean and standard deviation value from the training set 
    mean = tf.math.reduce_mean(train_data)
    std = tf.math.reduce_std(train_data)
    
    #Normalization formula (data - mean)/std
    train_data = (train_data - mean)/std
    test_data = (test_data - mean)/std
    
    #Convert the data into float
    train_data = tf.cast(train_data, dtype=tf.float32)
    test_data = tf.cast(test_data, dtype=tf.float32)
    #Convert labels into bool
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)
    #Separate the training data into normal and abnormal dataset 
    n_train_data = train_data[train_labels]    #normal training set
    an_train_data = train_data[~train_labels]   #abnormal training set

    #convert format ready for model training
    x_n_train = np.expand_dims(n_train_data,axis=2)
    x_an_train = np.expand_dims(an_train_data,axis=2)
    x_test = np.expand_dims(test_data,axis=2)
    return x_n_train, x_an_train, x_test, train_labels, test_labels

def build_autoencoder(x_train):
    """
    build autoencoder model
    :param x_train: training dateset
    :return: autoencoder model
    """
    n_steps = x_train.shape[1]
    n_features = x_train.shape[2]
    model = keras.Sequential(
        [
            #input layer
            layers.Input(shape=(n_steps, n_features)), 
            #Encoder 1D convolution layer
            layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',
                dilation_rate=1, activation="linear"),
            #Encoder Long Short-Term Memory layer 
            layers.LSTM(
                units=25, activation="tanh", name="lstm_1", return_sequences=False
            ),
            layers.RepeatVector(n_steps),
            #Decoder Long Short-Term Memory layer 
            layers.LSTM(
                units=25, activation="tanh", name="lstm_2", return_sequences=True
            ),
            #Decoder 1D convolution layer
            layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',
                dilation_rate=1, activation="linear"),
            layers.TimeDistributed(layers.Dense(x_train.shape[2], activation='linear'))
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    return model

def save_mae_graph(train_mae_loss,train_mae_loss_an,path):
    """
    Save MAE graph
    :param x_train: training dateset
    :return: autoencoder model
    """
    #plot size
    plt.subplots(figsize=(12,6))
    plt.hist(train_mae_loss, bins=50,label="normal", alpha=.6, color="blue")
    plt.hist(train_mae_loss_an, bins=50,label="abnormal", alpha=.6, color="red")
    plt.xlabel("Training MAE")
    plt.ylabel("No of samples")
    plt.legend()
    #set list of x-tick locations
    plt.xticks(np.arange(0, 1.6, 0.1)) 
    plt.savefig(path)


def print_stats(predictions, labels):
    """
    Calculat accuracy, precision and recall
    :param predictions: model prediction
    :param labels: label of dataset
    """
    print("Accuracy = {:.2%}".format(accuracy_score(labels, predictions)))
    print("Precision = {:.2%}".format(precision_score(labels, predictions)))
    print("Recall = {:.2%}".format(recall_score(labels, predictions)))

#load dataset
df = load_dataset('../data/training.csv')
#preproceing dataset
x_n_train, x_an_train, x_test, train_labels, test_labels = preprocessing(df)
#build autoencoder model using training set
autoencoder  = build_autoencoder(x_n_train)
#train the model
history = autoencoder.fit(x_n_train,
                          x_n_train,
                          epochs=150,
                          batch_size=128,
                          validation_split=0.1,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")
                        ],
)
#save the model
autoencoder.save('../models/autoencoder_model.keras') 
#predict on training set
x_n_train_pred = autoencoder.predict(x_n_train)
x_an_train_pred = autoencoder.predict(x_an_train)
#compute MAE LOSS for training set
n_train_mae_loss = np.mean(np.abs(x_n_train_pred - x_n_train), axis=1)
an_train_mae_loss = np.mean(np.abs(x_an_train_pred - x_an_train), axis=1)
#Gennerate & save MAE graph and decide threshold 
save_mae_graph(n_train_mae_loss,an_train_mae_loss,"../reports/training_MAE_loss.jpg")

print("Predicting on testing dataset")
#predict on testing set
x_test_pred = autoencoder.predict(x_test)
#compute mae on testing set
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
### this threshold is decided by mae graph, you may need to change this value everytime train the model
threshold = 0.3
#decide anomalies
normal_data = test_mae_loss < threshold
print("Calculating accuracy on testing dataset")
print_stats(normal_data,test_labels)

