import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

# Paths to the feature matrix and labels
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_labels.pt"

def one_hot_to_class_indices(one_hot_labels):
    """
    Convert one-hot encoded labels to class indices.

    Args:
        one_hot_labels (np.ndarray): One-hot encoded labels.

    Returns:
        np.ndarray: Class indices.
    """
    return np.argmax(one_hot_labels, axis=1)

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate the model using cross-validation.

    Args:
        model: The machine learning model to evaluate.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Class labels.
        cv (int): Number of cross-validation folds.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1-score).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Predict on validation data
        y_pred = model.predict(X_val)

        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
        metrics['f1_score'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

    return {metric: np.mean(scores) for metric, scores in metrics.items()}

if __name__ == "__main__":
    # Load feature matrix and labels
    print("Loading data...")
    X = torch.load(DATA_PATH).numpy()
    y_one_hot = torch.load(LABELS_PATH).numpy()
    print(f"Loaded data from {DATA_PATH}. Shape of X: {X.shape}")
    print(f"Loaded labels from {LABELS_PATH}. Shape of y: {y_one_hot.shape}")



    # Convert one-hot labels to class indices
    print("Converting one-hot encoded labels to class indices...")
    # Determine the format of y_one_hot and convert if necessary
    if y_one_hot.ndim == 1:
        y = y_one_hot  # Already in class index format
    else:
        y = np.argmax(y_one_hot, axis=1)  # Convert from one-hot

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label matrix shape: {y.shape}")
    assert X.shape[0] == y.shape[0], "Mismatch between feature matrix and label sizes."

    # Ensure data and labels have consistent lengths
    
    # Initialize a simple machine learning model (e.g., Logistic Regression)
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Evaluate the model
    print("Evaluating the model...")
    metrics = evaluate_model(model, X, y, cv=5)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Generate a classification report on the entire dataset
    print("Generating classification report on the entire dataset...")
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
