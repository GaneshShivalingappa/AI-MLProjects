import os
import pandas as pd
import numpy as np
import tensorflow as tf

def load_and_prepare_data():
    """Downloads, extracts, and prepares the Jena Climate dataset."""
    print("Downloading and preparing data...")
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)

    # The extraction process creates a directory with a known name.
    # Let's construct the path to the CSV file robustly.
    csv_path = os.path.join(os.path.dirname(zip_path), "jena_climate_2009_2016.csv")

    # Check if the default extracted file exists, if not, check the old path for compatibility
    if not os.path.exists(csv_path):
        csv_path = os.path.join(
            os.path.dirname(zip_path),
            "jena_climate_2009_2016_extracted",
            "jena_climate_2009_2016.csv"
        )

    df = pd.read_csv(csv_path)
    df = df[5::6]  # Subsample the data to one reading per hour.
    df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    temp = df['T (degC)']
    print("Data preparation complete.")
    return temp

def create_sequences(data, window_size=5):
    """Creates sequences and labels from time series data."""
    df_as_np = data.to_numpy()
    x, y = [], []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        x.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(x), np.array(y)