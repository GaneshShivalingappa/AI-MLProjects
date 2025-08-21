import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERROR

#Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Download and extract the dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

csv_path = os.path.join(
    os.path.dirname(zip_path), 
    "jena_climate_2009_2016_extracted", 
    "jena_climate_2009_2016.csv"
)

df = pd.read_csv(csv_path)

df = df[5::6]

df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

temp = df['T (degC)']

temp.plot()

def data_x_y(df, size = 5):
    df_as_np = df.to_numpy()
    x, y = [], []
    for i in range(len(df_as_np) - size):
       row = [[a] for a in df_as_np[i:i+size]]
       x.append(row)
       label = df_as_np[i + size]
       y.append(label)
    return np.array(x), np.array(y)

WindowSize = 5
X, y = data_x_y(temp, WindowSize)

X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(WindowSize, 1)))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.summary()

checkpoint = keras.callbacks.ModelCheckpoint("bestModel/best_model.keras", save_best_only=True)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[checkpoint])
