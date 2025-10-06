import numpy as np
import os

CONFIG = {
    "output_dir": "scenarios", 
    "scenario_length_seconds": 30,
    "signal_length": 128, 
    "sample_rate": 1e6, 
}

DEVICES = {
    "friendly_drone": {
        "fingerprint": {"phase_offset": 0.1, "frequency_drift": 50, "snr_db": 25},
        "label": 0
    },
    "hostile_drone": {
        "fingerprint": {"phase_offset": -0.15, "frequency_drift": -75, "snr_db": 15},
        "label": 1
    },
    "iot_device": {
        "fingerprint": {"phase_offset": 0.0, "frequency_drift": 10, "snr_db": 20},
        "label": 2
    },
    "jammer": { 
        "fingerprint": {"snr_db": 5}, 
        "label": 3
    }
}

def generate_base_signal(length, device_name):
    if device_name == "jammer":
        return np.random.randn(length) + 1j * np.random.randn(length)
    num_symbols = length // 2
    symbols = np.random.randint(0, 4, num_symbols)
    qpsk_map = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
    iq_data = np.array([qpsk_map[s] for s in symbols]) / np.sqrt(2)
    return np.repeat(iq_data, 2)

def apply_fingerprint(signal, fingerprint):
    if "phase_offset" in fingerprint:
        signal = signal * np.exp(1j * fingerprint["phase_offset"])
    if "frequency_drift" in fingerprint:
        t = np.arange(len(signal)) / CONFIG["sample_rate"]
        freq_drift_term = np.exp(1j * 2 * np.pi * fingerprint["frequency_drift"] * t)
        signal = signal * freq_drift_term
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (fingerprint["snr_db"] / 10)
    noise_power = signal_power / snr_linear
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * np.sqrt(noise_power / 2)
    return signal + noise

def create_scenario(name, events):
    print(f"Creating scenario: {name}")
    samples_per_second = int(CONFIG["sample_rate"] / CONFIG["signal_length"])
    total_samples = CONFIG["scenario_length_seconds"] * samples_per_second
    scenario_timeline = np.zeros((total_samples, CONFIG["signal_length"]), dtype=np.complex64)
    for event in events:
        device_name = event["device"]
        start_sec = event["start"]
        end_sec = event["end"]
        print(f"  - Adding {device_name} from {start_sec}s to {end_sec}s")
        start_sample = start_sec * samples_per_second
        end_sample = end_sec * samples_per_second
        for i in range(start_sample, end_sample):
            base_signal = generate_base_signal(CONFIG["signal_length"], device_name)
            fingerprinted_signal = apply_fingerprint(base_signal, DEVICES[device_name]["fingerprint"])
            scenario_timeline[i, :] += fingerprinted_signal
    output_path = os.path.join(CONFIG["output_dir"], f"{name}.npy")
    np.save(output_path, scenario_timeline)
    print(f"Scenario saved to {output_path}")



def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    print("\nGenerating test scenarios...")
    scenario_1_events = [{"device": "iot_device", "start": 0, "end": 30}, {"device": "friendly_drone", "start": 5, "end": 25}, {"device": "hostile_drone", "start": 18, "end": 21}]
    create_scenario("1_stealth_incursion", scenario_1_events)
    scenario_2_events = [{"device": "friendly_drone", "start": 0, "end": 30}, {"device": "jammer", "start": 10, "end": 20}]
    create_scenario("2_disruption_attack", scenario_2_events)
    print("\nScenario generation complete!")

    

if __name__ == "__main__":
    main()
