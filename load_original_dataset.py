import os
import torch
import numpy as np

# Dataset directory
DATA_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\Bon"
# Output directory for processed data
OUTPUT_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB"

# Mapping of subdirectories to labels
LABEL_MAPPING = {"f": 0, "n": 1, "o": 2, "s": 3, "z": 4}

def load_txt_file(file_path):
    """
    Loads a single .txt file into a NumPy array.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        np.ndarray: Loaded data.
    """
    return np.loadtxt(file_path)

def load_dataset_from_directory(data_dir):
    """
    Loads the dataset from a directory structure.

    Args:
        data_dir (str): Root directory containing subdirectories for each class.

    Returns:
        torch.Tensor, torch.Tensor: Data tensor and labels tensor.
    """
    data = []
    labels = []

    for subdir, label in LABEL_MAPPING.items():
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Warning: Directory {subdir_path} does not exist.")
            continue

        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        sample = load_txt_file(file_path)
                        data.append(sample)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, labels

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    data, labels = load_dataset_from_directory(DATA_DIR)

    # Display dataset info
    print(f"Data Shape: {data.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Class Distribution: {torch.bincount(labels)}")

    # Save dataset
    torch.save(data, os.path.join(OUTPUT_DIR, "original_data.pt"))
    torch.save(labels, os.path.join(OUTPUT_DIR, "original_labels.pt"))
    print(f"Dataset saved to {OUTPUT_DIR}.")
