import torch
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_labels.pt"
OUTPUT_DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\balanced_data.pt"
OUTPUT_LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\balanced_labels.pt"

def balance_data_with_smote(data, labels):
    """
    Balances the dataset using SMOTE.

    Args:
        data (np.ndarray): Feature matrix.
        labels (np.ndarray): Labels array.

    Returns:
        np.ndarray, np.ndarray: Balanced feature matrix and labels.
    """
    print("Original class distribution:", Counter(labels))
    smote = SMOTE(random_state=42)
    balanced_data, balanced_labels = smote.fit_resample(data, labels)
    print("Balanced class distribution:", Counter(balanced_labels))
    return balanced_data, balanced_labels

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = torch.load(DATA_PATH).numpy()
    labels = torch.load(LABELS_PATH).numpy()

    # Balance data
    print("Balancing dataset using SMOTE...")
    balanced_data, balanced_labels = balance_data_with_smote(data, labels)

    # Save balanced data
    print("Saving balanced dataset...")
    torch.save(torch.tensor(balanced_data, dtype=torch.float32), OUTPUT_DATA_PATH)
    torch.save(torch.tensor(balanced_labels, dtype=torch.long), OUTPUT_LABELS_PATH)
    print(f"Balanced data saved to {OUTPUT_DATA_PATH} and {OUTPUT_LABELS_PATH}")
