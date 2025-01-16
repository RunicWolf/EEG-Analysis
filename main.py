from data_loader import load_data, save_data, split_and_save_dataset
from preprocessing import handle_missing_values, handle_outliers, normalize_data
from augmentation import augment_and_label
import torch

from feature_engineering import (
    one_hot_encode_labels,
    apply_pca,
    select_top_features,
    remove_constant_features,
    extract_features
)
from sklearn.model_selection import train_test_split
import numpy as np
import os

if __name__ == "__main__":
    DATA_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\Bon"
    OUTPUT_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB"

    print("Loading augmented data...")
    data = load_data(os.path.join(OUTPUT_DIR, "augmented_data.pt"))
    labels = load_data(os.path.join(OUTPUT_DIR, "augmented_labels.pt"))

    print("Normalizing data...")
    data = normalize_data(data, method="zscore")

    print("Handling outliers...")
    data, labels = handle_outliers(data, labels, method="zscore", threshold=3)

    # Balance the dataset
    print("Balancing the dataset...")
    balanced_data, balanced_labels = augment_and_label(data, labels, num_classes=3, augment_factor=2)
    print(f"Balanced label distribution: {np.unique(balanced_labels, return_counts=True)}")

    print("Extracting features...")
    features = extract_features(balanced_data)

    print(f"Feature variances before removal: {np.var(features, axis=0)}")
    print("Removing low-variance features...")
    features = remove_constant_features(features, threshold=0.01)
    print(f"Feature variances after removal: {np.var(features, axis=0)}")

    print("One-hot encoding labels...")
    one_hot_labels, flat_labels = one_hot_encode_labels(balanced_labels)

    print(f"Label distribution: {np.unique(flat_labels, return_counts=True)}")

    print("Applying PCA...")
    features = apply_pca(features)

    print(f"Feature variances before SelectKBest: {np.var(features, axis=0)}")
    print("Selecting top features...")
    k = min(5, features.shape[1])
    features = select_top_features(features, flat_labels, k=k)

    print(f"Augmented Data Shape: {balanced_data.shape}")
    print(f"Augmented Labels Shape: {balanced_labels.shape}")

    # Splitting and saving augmented dataset
    print("Splitting and saving augmented dataset...")
    split_and_save_dataset(
    data=torch.tensor(balanced_data, dtype=torch.float32),
    labels=torch.tensor(balanced_labels, dtype=torch.long),
    output_dir=OUTPUT_DIR,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)

    print(f"Data split and saved to {OUTPUT_DIR}.")
