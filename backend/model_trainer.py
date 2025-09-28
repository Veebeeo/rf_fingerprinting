import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


DATASET_DIR = "dataset"

MODEL_SAVE_PATH = "rf_fingerprint_model.keras" 
TEST_SPLIT_SIZE = 0.2

def load_data(dataset_path):
    """Loads the I/Q data from our generated dataset directory."""
    print(f"Loading data from '{dataset_path}'...")
    all_signals = []
    all_labels = []
    
    device_names = sorted(os.listdir(dataset_path))
    
    for device_name in device_names:
        device_dir = os.path.join(dataset_path, device_name)
        if not os.path.isdir(device_dir):
            continue
        
        for file_name in os.listdir(device_dir):
            if file_name.endswith(".npy"):
                file_path = os.path.join(device_dir, file_name)
                signal = np.load(file_path)
                all_signals.append(signal)
                all_labels.append(device_name)
    
    print(f"Loaded {len(all_signals)} total samples.")
    return np.array(all_signals), np.array(all_labels), device_names

def preprocess_data(signals, labels):
    """Prepares the data for training the CNN."""
    print("Preprocessing data...")
    signals_real_imag = np.stack([signals.real, signals.imag], axis=-1)
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_labels = onehot_encoder.fit_transform(integer_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        signals_real_imag, onehot_labels, test_size=TEST_SPLIT_SIZE, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes):
    """Defines the 1D Convolutional Neural Network architecture."""
    print("Building CNN model...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=8, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def main():
    """Main function to run the entire training pipeline."""
    signals, labels, device_names = load_data(DATASET_DIR)
    num_classes = len(device_names)
    
    X_train, X_test, y_train, y_test = preprocess_data(signals, labels)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes)
    
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    print(f"Saving trained model to '{MODEL_SAVE_PATH}'...")
    
    model.save(MODEL_SAVE_PATH) 
    print("Model saved successfully.")

if __name__ == "__main__":
    main()

