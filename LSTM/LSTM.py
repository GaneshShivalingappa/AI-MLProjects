#Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

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

# Display the first few rows of the dataframe
print(df.head())
