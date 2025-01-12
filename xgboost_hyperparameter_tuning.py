import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from itertools import product

# Load the dataset
file_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\UKB_EEG_extracted_features_cuda.csv"
data = pd.read_csv(file_path)

# Select top 6 features based on previous analysis
selected_features = ['Alpha_Power', 'Zero-Crossing Rate', 'Beta_Power', 
                     'Sample_Entropy', 'Theta_Power', 'Delta_Power']

X = data[selected_features]  # Use only selected features
y = data['Set']  # Target column

# Encode string labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert 'F', 'N', etc. to 0, 1, 2, ...

# Define parameter grid for manual search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Generate all combinations of parameters
param_combinations = list(product(
    param_grid['n_estimators'],
    param_grid['learning_rate'],
    param_grid['max_depth'],
    param_grid['min_child_weight'],
    param_grid['subsample'],
    param_grid['colsample_bytree']
))

# Initialize KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Manual grid search
best_score = 0
best_params = None

print("\n=== Starting Manual Hyperparameter Tuning with GPU ===")
for params in param_combinations:
    n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree = params

    # Define the model with current parameters and GPU
    xgb = XGBClassifier(
        random_state=42, eval_metric='mlogloss',
        tree_method='hist',  # Use 'hist' for GPU support
        device='cuda',       # Specify CUDA for GPU acceleration
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    # Perform cross-validation
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Train and evaluate
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

    # Compute mean accuracy
    mean_score = sum(scores) / len(scores)

    # Update best parameters if the current model is better
    if mean_score > best_score:
        best_score = mean_score
        best_params = params

    print(f"Params: {params}, Mean CV Accuracy: {mean_score:.4f}")

# Display the best parameters and score
print("\n=== Best Hyperparameters ===")
print(f"n_estimators: {best_params[0]}, learning_rate: {best_params[1]}, max_depth: {best_params[2]},")
print(f"min_child_weight: {best_params[3]}, subsample: {best_params[4]}, colsample_bytree: {best_params[5]}")
print(f"Best Mean CV Accuracy: {best_score:.4f}")
