from data_loader import load_data, save_data
from preprocessing import handle_missing_values, handle_outliers, normalize_data
from augmentation import augment_and_label
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

    print("Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, one_hot_labels, test_size=0.3, stratify=flat_labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=flat_labels[:len(y_temp)], random_state=42
    )

    print("Saving split data...")
    save_data(X_train, os.path.join(OUTPUT_DIR, "train.pt"))
    save_data(X_val, os.path.join(OUTPUT_DIR, "val.pt"))
    save_data(X_test, os.path.join(OUTPUT_DIR, "test.pt"))
    save_data(y_train, os.path.join(OUTPUT_DIR, "train_labels.pt"))
    save_data(y_val, os.path.join(OUTPUT_DIR, "val_labels.pt"))
    save_data(y_test, os.path.join(OUTPUT_DIR, "test_labels.pt"))
    print(f"Data split and saved to {OUTPUT_DIR}.")
