

import scipy.io

# Path to one of your specific .mat files
file_path = "BMIS_EEG_DATA/data/mat_data/subject_1/S1_R1_G1.mat"

# Load the file
mat_data = scipy.io.loadmat(file_path)

# Filter out the default MATLAB metadata keys
clean_keys = [key for key in mat_data.keys() if not key.startswith('__')]

print("--- FILE ANALYSIS ---")
print(f"Variables found in file: {clean_keys}\n")

# Display the matrix size and data type for each variable
for key in clean_keys:
    data_array = mat_data[key]
    print(f"Variable Name: '{key}'")
    print(f"Data Type: {type(data_array)}")
    if hasattr(data_array, 'shape'):
        print(f"Matrix Shape: {data_array.shape}")
    print("-" * 30)
