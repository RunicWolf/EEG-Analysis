import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Paths to the data
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_labels.pt"

# Load data
def load_data(data_path, labels_path):
    data = torch.load(data_path).numpy()
    labels = torch.load(labels_path).numpy()
    return data, labels

# Label Encoding Function
def encode_labels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y)

# Stratified K-Fold Cross-Validation
def cross_validate_model(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Standard deviation of accuracy: {std_accuracy}")
    return mean_accuracy, std_accuracy

if __name__ == "__main__":
    # Load data
    X, y = load_data(DATA_PATH, LABELS_PATH)
    y = encode_labels(y)

    print(f"Loaded feature matrix of shape: {X.shape}")
    print(f"Loaded label matrix of shape: {y.shape}")

    # Initialize Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    cross_validate_model(rf_model, X, y)
