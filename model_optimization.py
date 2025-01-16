import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths to the augmented data and labels
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_labels.pt"

# Load augmented data and labels
def load_data(data_path, labels_path):
    data = torch.load(data_path).numpy()
    labels = torch.load(labels_path).numpy()
    return data, labels

# Label Encoding Function
def encode_labels(y):
    """
    Encode labels to integer values.
    """
    if len(y.shape) == 2:  # If labels are one-hot encoded
        y = np.argmax(y, axis=1)  # Convert to integers

    # Ensure labels are continuous starting from 0
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    return y_encoded

# Load data
X, y = load_data(DATA_PATH, LABELS_PATH)
print(f"Loaded feature matrix of shape: {X.shape}")
print(f"Loaded label matrix of shape: {y.shape}")

# Convert labels to integers
y = encode_labels(y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Standardize the features (this is important for many machine learning models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Best score for Random Forest: {grid_search_rf.best_score_}")

# XGBoost Model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Hyperparameter tuning with RandomizedSearchCV
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0]
}
random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist_xgb, cv=5, n_jobs=-1, n_iter=10)
random_search_xgb.fit(X_train, y_train)

print(f"Best parameters for XGBoost: {random_search_xgb.best_params_}")
print(f"Best score for XGBoost: {random_search_xgb.best_score_}")

# LightGBM Model
lgb_model = lgb.LGBMClassifier(random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist_lgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [31, 50, 100],
    'max_depth': [3, 6, 10]
}
random_search_lgb = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_dist_lgb, cv=5, n_jobs=-1, n_iter=10)
random_search_lgb.fit(X_train, y_train)

print(f"Best parameters for LightGBM: {random_search_lgb.best_params_}")
print(f"Best score for LightGBM: {random_search_lgb.best_score_}")

# Model evaluation on test data
best_rf_model = grid_search_rf.best_estimator_
rf_test_accuracy = best_rf_model.score(X_test, y_test)
print(f"Optimized Random Forest Test Accuracy: {rf_test_accuracy}")

best_xgb_model = random_search_xgb.best_estimator_
xgb_test_accuracy = best_xgb_model.score(X_test, y_test)
print(f"Optimized XGBoost Test Accuracy: {xgb_test_accuracy}")

best_lgb_model = random_search_lgb.best_estimator_
lgb_test_accuracy = best_lgb_model.score(X_test, y_test)
print(f"Optimized LightGBM Test Accuracy: {lgb_test_accuracy}")
