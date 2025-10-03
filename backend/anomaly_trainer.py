import numpy as np
import tensorflow as tf
import pickle
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from sklearn.model_selection import train_test_split

REAL_WORLD_DATASET_PATH = "RML2016.10a_dict.pkl"
ANOMALY_MODEL_SAVE_PATH = "anomaly_detector.keras"
NORMAL_CLASSES = ['GFSK', 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'CPFSK', 'PAM4']

def load_normal_data(path):
    print(f"Loading 'normal' data for anomaly detector exclusively from real-world dataset: '{path}'...")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    signals = []
    for modulation, snr in data.keys():
        if modulation in NORMAL_CLASSES:
            signals.extend(data[(modulation, snr)])
    print(f"Loaded {len(signals)} 'normal' samples from the real-world dataset.")
    return np.array(signals)

def preprocess_anomaly_data(signals):
    print("Preprocessing anomaly data...")
    signals = np.transpose(signals, (0, 2, 1))
    X_train, X_test = train_test_split(signals, test_size=0.2, random_state=42)
    return X_train, X_test

def build_autoencoder(input_shape):
    print("Building Autoencoder model...")
    input_sig = Input(shape=input_shape)
    x = Conv1D(16, 3, activation='relu', padding='same')(input_sig)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(2, 3, activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_sig, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    return autoencoder

def main():
    signals = load_normal_data(REAL_WORLD_DATASET_PATH)
    X_train, X_test = preprocess_anomaly_data(signals)
    input_shape = (X_train.shape[1], X_train.shape[2])
    autoencoder = build_autoencoder(input_shape)
    print("\nStarting autoencoder training on real-world data only...")
    autoencoder.fit(X_train, X_train,
                    epochs=20,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))
    print(f"\nSaving anomaly detector model to '{ANOMALY_MODEL_SAVE_PATH}'...")
    autoencoder.save(ANOMALY_MODEL_SAVE_PATH)
    print("Anomaly detector saved successfully.")

if __name__ == "__main__":
    main()
