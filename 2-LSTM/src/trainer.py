import os
from tensorflow import keras

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.0001):
    """Compiles and trains the model."""
    checkpoint_path = "bestModel/best_model_1DCNN.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    cp1 = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.RootMeanSquaredError()])

    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[cp1])
    print("Model training complete.")
    return history