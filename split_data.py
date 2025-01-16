import torch
from sklearn.model_selection import train_test_split
import os

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\fixed_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\fixed_labels.pt"

OUTPUT_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB"

def split_data(data, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits data into training, validation, and test sets.

    Args:
        data (ndarray): Input features.
        labels (ndarray): Labels for the data.
        test_size (float): Proportion of data for testing.
        val_size (float): Proportion of data for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: Train, validation, and test sets.
    """
    # Split into train and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=test_size + val_size, stratify=labels, random_state=random_state
    )

    # Split temp into validation and test sets
    test_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Load data and labels
    print("Loading data...")
    data = torch.load(DATA_PATH).numpy()
    labels = torch.load(LABELS_PATH).numpy()

    print("Splitting data into training, validation, and testing sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, labels)

    print(f"Train Data Shape: {X_train.shape}, Train Labels Shape: {y_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}, Validation Labels Shape: {y_val.shape}")
    print(f"Test Data Shape: {X_test.shape}, Test Labels Shape: {y_test.shape}")

    # Save the splits
    print("Saving split data...")
    torch.save(torch.tensor(X_train, dtype=torch.float32), os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(torch.tensor(y_train, dtype=torch.long), os.path.join(OUTPUT_DIR, "train_labels.pt"))
    torch.save(torch.tensor(X_val, dtype=torch.float32), os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save(torch.tensor(y_val, dtype=torch.long), os.path.join(OUTPUT_DIR, "val_labels.pt"))
    torch.save(torch.tensor(X_test, dtype=torch.float32), os.path.join(OUTPUT_DIR, "test.pt"))
    torch.save(torch.tensor(y_test, dtype=torch.long), os.path.join(OUTPUT_DIR, "test_labels.pt"))

    print(f"Data splits saved to {OUTPUT_DIR}.")
