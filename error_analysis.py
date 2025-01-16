import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Model evaluation and error analysis
def error_analysis(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()

    # Save misclassified examples
    misclassified = np.where(y_test != y_pred)
    np.save('misclassified_examples.npy', X_test[misclassified])
    np.save('misclassified_labels.npy', y_test[misclassified])
    print(f"Misclassified examples saved.")

if __name__ == "__main__":
    # Load data
    X, y = load_data(DATA_PATH, LABELS_PATH)
    y = encode_labels(y)

    print(f"Loaded feature matrix of shape: {X.shape}")
    print(f"Loaded label matrix of shape: {y.shape}")

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Perform error analysis
    error_analysis(rf_model, X_test, y_test)
