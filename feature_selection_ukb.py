import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt  # Ensure this is imported

# Load the dataset
file_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\UKB_EEG_extracted_features_cuda.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['Set', 'File', 'Channel'])  # Feature columns
y = data['Set']  # Target column

# Verify dataset structure
print("=== Dataset Structure Check ===")
print("Feature columns in X:", X.columns.tolist())
print("Number of features in X:", X.shape[1])
print("Number of target classes:", y.nunique())
if X.shape[1] != 12:
    print(f"Warning: Expected 12 features but found {X.shape[1]} features.")
    print("Please verify the dataset preprocessing step.")
else:
    print("Dataset structure is as expected.\n")

# Train a Random Forest Model
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X, y)

# Compute Permutation Importance
print("Computing Permutation Importance...")
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

# Display Results
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance Mean": perm_importance.importances_mean,
    "Importance STD": perm_importance.importances_std
}).sort_values(by="Importance Mean", ascending=False)

print("\n=== Permutation Importance Results ===")
print(importance_df)

# Visualize Results
plt.figure(figsize=(10, 6))
plt.bar(
    x=importance_df["Feature"],
    height=importance_df["Importance Mean"],
    yerr=importance_df["Importance STD"],
    capsize=5
)
plt.title("Permutation Importance")
plt.xlabel("Features")
plt.ylabel("Mean Decrease in Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
