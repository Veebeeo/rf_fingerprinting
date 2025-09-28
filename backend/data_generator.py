import numpy as np
import os

CONFIG = {
    "output_dir": "dataset",
    "num_samples_per_device": 1000, 
    "signal_length": 1024,
    "sample_rate": 1e6, 
}

# --- Device Fingerprint Library ---
# This is the heart of our simulation. Each device has a unique set of imperfections.
DEVICES = {
    "friendly_drone": {
        "fingerprint": {
            "phase_offset": 0.1,   # Slight, consistent phase error
            "frequency_drift": 50, # 50 Hz frequency drift
            "snr_db": 25           # Signal-to-Noise Ratio in dB (clean signal)
        },
        "label": 0
    },
    "hostile_drone": {
        "fingerprint": {
            "phase_offset": -0.15, # Different phase error
            "frequency_drift": -75, # Drifts in the opposite direction
            "snr_db": 15            # Noisier signal (lower SNR)
        },
        "label": 1
    },
    "iot_device": {
        "fingerprint": {
            "phase_offset": 0.0,    # No phase error
            "frequency_drift": 10,  # Very low drift
            "snr_db": 20            # Moderately clean signal
        },
        "label": 2
    },
}

def generate_base_signal(length):
    """Generates a simple QPSK (Quadrature Phase-Shift Keying) signal."""
    num_symbols = length // 2
    # Generate random symbols (0, 1, 2, 3)
    symbols = np.random.randint(0, 4, num_symbols)
    # Map symbols to complex numbers on the unit circle
    qpsk_map = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
    iq_data = np.array([qpsk_map[s] for s in symbols]) / np.sqrt(2) # Normalize power
    # Repeat samples to match the required signal length (simple pulse shaping)
    iq_data = np.repeat(iq_data, 2)
    return iq_data

def apply_fingerprint(signal, fingerprint):
    """Applies the unique hardware imperfections to the base signal."""
    # 1. Apply Phase Offset
    signal_with_phase_offset = signal * np.exp(1j * fingerprint["phase_offset"])
    
    # 2. Apply Frequency Drift
    t = np.arange(len(signal)) / CONFIG["sample_rate"]
    freq_drift_term = np.exp(1j * 2 * np.pi * fingerprint["frequency_drift"] * t)
    signal_with_drift = signal_with_phase_offset * freq_drift_term
    
    # 3. Add Gaussian Noise (simulating SNR)
    signal_power = np.mean(np.abs(signal_with_drift) ** 2)
    snr_linear = 10 ** (fingerprint["snr_db"] / 10)
    noise_power = signal_power / snr_linear
    
    
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
    noise = noise_real + 1j * noise_imag
    
    final_signal = signal_with_drift + noise
    return final_signal

def main():
    
    print("Starting synthetic dataset generation...")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    
    for device_name, device_info in DEVICES.items():
        print(f"  Generating data for: {device_name}")
        device_dir = os.path.join(CONFIG["output_dir"], device_name)
        os.makedirs(device_dir, exist_ok=True)
        
        
        for i in range(CONFIG["num_samples_per_device"]):
            base_signal = generate_base_signal(CONFIG["signal_length"])
            fingerprinted_signal = apply_fingerprint(base_signal, device_info["fingerprint"])
            
            
            file_path = os.path.join(device_dir, f"sample_{i:04d}.npy")
            np.save(file_path, fingerprinted_signal)
            
    print("Dataset generation complete!")
    print(f"Data saved in '{CONFIG['output_dir']}' directory.")

if __name__ == "__main__":
    main()