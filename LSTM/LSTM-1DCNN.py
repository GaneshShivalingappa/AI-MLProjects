import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERROR

# Import project modules
from src.data_processing import load_and_prepare_data, create_sequences
from src.model_builder import build_model
from src.trainer import train_model

def main():
    """Main function to run the ML pipeline."""
    # --- 1. Configuration ---
    WINDOW_SIZE = 5
    TRAIN_SPLIT = 60000
    VAL_SPLIT = 65000
    EPOCHS = 10

    # --- 2. Data Loading and Preprocessing ---
    temp_series = load_and_prepare_data()
    X, y = create_sequences(temp_series, WINDOW_SIZE)

    X_train, y_train = X[:TRAIN_SPLIT], y[:TRAIN_SPLIT]
    X_val, y_val = X[TRAIN_SPLIT:VAL_SPLIT], y[TRAIN_SPLIT:VAL_SPLIT]
    X_test, y_test = X[VAL_SPLIT:], y[VAL_SPLIT:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --- 3. Model Building and Training ---
    # The input shape for the model is (window_size, num_features)
    model = build_model(input_shape=(WINDOW_SIZE, 1))
    
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS)
    
    print("\nPipeline finished successfully.")
    # You can now use the 'history' object or load the best model for evaluation.


if __name__ == "__main__":
    main()
