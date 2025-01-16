import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Paths to the augmented data and labels
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_labels.pt"

# Load augmented data and labels
def load_data(data_path, labels_path):
    data = torch.load(data_path).numpy()
    labels = torch.load(labels_path).numpy()
    return data, labels

# 1. Model Optimization (Random Forest, XGBoost, LightGBM)
# Hyperparameter tuning for Random Forest
rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 2. Cross-Validation Enhancement
def cross_validate_model(model, X, y, folds=10):
    stratified_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, val_index in stratified_kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracies.append(accuracy_score(y_val, y_pred))
    
    print(f"Mean accuracy: {np.mean(accuracies)}")
    print(f"Standard deviation of accuracy: {np.std(accuracies)}")
    return model

# 3. Error Analysis (Confusion Matrix and Misclassified Examples)
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def save_misclassified_examples(X_test, y_test, y_pred, file_path="misclassified_data.npy"):
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_data = X_test[misclassified_indices]
    misclassified_labels = y_test[misclassified_indices]
    np.save(file_path, {"data": misclassified_data, "labels": misclassified_labels})

# 4. Experiment with Deep Learning (MLP Model)
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def train_mlp_model(X_train, y_train, input_dim, hidden_dim=64, output_dim=None, num_epochs=10):
    model = MLPModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Ensure labels are in the correct range (0 to num_classes-1)
    y_train = y_train.astype(int)  # Ensure labels are integers
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return model

def main():
    # Load data
    X, y = load_data(DATA_PATH, LABELS_PATH)
    
    # Check number of unique classes in labels
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    
    # Ensure labels are in the correct range (0 to num_classes-1)
    y = y - np.min(y)  # Make sure labels start from 0
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model Optimization (Random Forest, XGBoost, LightGBM) and Cross-validation
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = cross_validate_model(grid_search_rf.best_estimator_, X_train, y_train)
    y_pred_rf = best_rf_model.predict(X_test)
    
    # Error analysis
    plot_confusion_matrix(y_test, y_pred_rf, classes=np.unique(y_test))
    save_misclassified_examples(X_test, y_test, y_pred_rf)
    
    # Neural Network (MLP)
    model = train_mlp_model(X_train, y_train, input_dim=X_train.shape[1], output_dim=num_classes)

if __name__ == "__main__":
    main()


