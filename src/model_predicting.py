from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt
import training_var

colnames=['timestamp','value'] 
df_test = pd.read_csv('../data/test.csv',names=colnames, parse_dates=True, index_col="timestamp")
model = keras.models.load_model('../res/model.keras')

TIME_STEPS = int(24*7)


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

df_test_value = (df_test - training_var.training_mean) / training_var.training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()

# Create sequences from test values.
x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("Testing MAE")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > training_var.threshold


anomalous_data_indices_raw = np.where(anomalies)[0]
anomalous_data_indices_raw = [x+TIME_STEPS/2 for x in anomalous_data_indices_raw]
anomalous_data_indices = []
for data_idx in anomalous_data_indices_raw:
    if set(range(int(data_idx)-int(TIME_STEPS/4),int(data_idx)+int(TIME_STEPS/4))).issubset(set(anomalous_data_indices_raw)):
        anomalous_data_indices.append(int(data_idx))
        

print("Number of anomaly samples: ", len(anomalous_data_indices))
print("Indices of anomaly samples: ", anomalous_data_indices)

