import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

# Verify reduced feature set and encoded labels
print("=== Reduced Dataset Structure ===")
print("Selected features:", X.columns.tolist())
print("Number of features in X:", X.shape[1])
print("Number of target classes:", len(label_encoder.classes_))
print("Encoded classes:", list(label_encoder.classes_))

# Train and evaluate XGBoost using KFold cross-validation
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

print("\n=== Manual Cross-Validation Results ===")
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Train on training set
    xgb.fit(X_train, y_train)

    # Predict on test set
    y_pred = xgb.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    print(f"Fold {fold + 1}: Accuracy = {acc:.4f}")

print("\nOverall Accuracy:")
print(f"Mean Accuracy: {sum(scores) / len(scores):.4f}")
print(f"Standard Deviation: {pd.Series(scores).std():.4f}")

# Compute Permutation Importance for reduced features
print("\nComputing Permutation Importance (Reduced Features with XGBoost)...")
perm_importance = permutation_importance(xgb, X, y_encoded, n_repeats=10, random_state=42)

# Display Results
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance Mean": perm_importance.importances_mean,
    "Importance STD": perm_importance.importances_std
}).sort_values(by="Importance Mean", ascending=False)

print("\n=== Permutation Importance Results (Reduced Features with XGBoost) ===")
print(importance_df)

# Visualize Permutation Importance
plt.figure(figsize=(8, 5))
plt.bar(
    x=importance_df["Feature"],
    height=importance_df["Importance Mean"],
    yerr=importance_df["Importance STD"],
    capsize=5
)
plt.title("Permutation Importance (Reduced Features with XGBoost)")
plt.xlabel("Features")
plt.ylabel("Mean Decrease in Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
