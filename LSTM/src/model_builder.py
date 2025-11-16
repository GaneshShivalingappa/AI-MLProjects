import tensorflow as tf
from tensorflow import keras

def build_model(input_shape):
    """Builds and returns the 1D CNN model."""
    print("Building model...")
    model = tf.keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    model.summary()
    return model