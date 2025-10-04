import numpy as np
import os


source_directory = 'frontend/public'


original_files = [
    '1_stealth_incursion.npy',
    '2_disruption_attack.npy',
    'real_world_QPSK_test.npy'
]


num_points_to_keep = 150000

print("Starting to create smaller demo samples...")

for filename in original_files:
    original_file_path = os.path.join(source_directory, filename)
    new_filename = f"demo_{filename}"
    new_file_path = os.path.join(source_directory, new_filename)

    try:
        print(f"\nProcessing '{filename}'...")
        original_data = np.load(original_file_path, allow_pickle=True)
     
        points = min(num_points_to_keep, len(original_data))
        small_data_slice = original_data[:points]
        
        print(f"Saving smaller file with {len(small_data_slice)} data points to '{new_file_path}'...")
        np.save(new_file_path, small_data_slice)
        
    except FileNotFoundError:
        print(f"--> ERROR: The file '{original_file_path}' was not found. Skipping.")
    except Exception as e:
        print(f"--> An error occurred while processing {filename}: {e}")

print("\nSuccessfully created all smaller demo files!")
