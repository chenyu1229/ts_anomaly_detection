from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
import sys



def read_dataset(file):
    df = pd.read_csv(file, parse_dates=True, index_col="timestamp",header=0)
    return df

# Generated training sequences for use in the model.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def pre_processing(df,time_steps):
    df = df.dropna()
    training_mean = df.mean()
    training_std = df.std()
    df_training_value = (df - training_mean) / training_std
    print("Number of training samples:", len(df_training_value))
    x_train = create_sequences(df_training_value,time_steps)
    print("Training input shape: ", x_train.shape)
    return x_train, training_mean, training_std


# x_train = create_sequences(df_training_value)
# print("Training input shape: ", x_train.shape)
def build_model(x_train):
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



def save_res(train_mae_loss,training_mean,training_std):
    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Training MAE")
    plt.ylabel("No of samples")
    plt.savefig("../res/training_MAE_loss.jpg")

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    # mesg = "threshold="+str(threshold)+"\n"+"training_mean="+str(float(training_mean.iloc[0]))+"\n"+"training_std="+str(float(training_std.iloc[0]))+"\n"+"time_steps="+str(int(TIME_STEPS))
    mesg = "threshold="+str(threshold)+"\n"+"training_mean="+str(list(training_mean))+"\n"+"training_std="+str(list(training_std))+"\n"+"time_steps="+str(int(TIME_STEPS))
    print("Reconstruction error threshold: ", threshold)

    model.save('../res/model.keras') 
    with open("training_var.py","w") as f:
        f.write(mesg)
    print("Model Saved")


TIME_STEPS = int(24*7)
# if len(sys.argv) > 1:
#     TIME_STEPS = int(sys.argv[1])
# print (TIME_STEPS)
df = read_dataset('../data/train_multi.csv')
x_train, training_mean, training_std = pre_processing(df,TIME_STEPS)
model  = build_model(x_train)

history = model.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")
    ],
)

print("Calculating Training MAE Loss")
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.mean(np.abs(x_train_pred - x_train), axis=1), axis=1)
save_res(train_mae_loss,training_mean,training_std)





