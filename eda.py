import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_labels.pt"
OUTPUT_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\correlations"

# Device for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_cleaned_data(data_path, labels_path, device):
    """
    Loads the cleaned dataset and corresponding labels.

    Args:
        data_path (str): Path to the cleaned data file.
        labels_path (str): Path to the labels file.
        device (torch.device): Device for loading the data.

    Returns:
        pd.DataFrame: Data as a DataFrame.
        pd.Series: Labels as a Series.
    """
    data = torch.load(data_path, map_location=device).cpu().numpy()
    labels = torch.load(labels_path, map_location=device).cpu().numpy()
    df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(data.shape[1])])
    df['Label'] = labels
    return df

def analyze_correlations(df, output_dir):
    """
    Analyzes correlations between features and saves the heatmap.

    Args:
        df (pd.DataFrame): DataFrame containing features and labels.
        output_dir (str): Directory to save the correlation heatmap.
    """
    print("Analyzing feature correlations...")
    corr_matrix = df.drop(columns=['Label']).corr()

    # Save correlation matrix
    os.makedirs(output_dir, exist_ok=True)
    corr_matrix_path = os.path.join(output_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_matrix_path)
    print(f"Correlation matrix saved to {corr_matrix_path}")

    # Plot and save heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title("Feature Correlation Matrix")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    # Load data
    df = load_cleaned_data(DATA_PATH, LABELS_PATH, device)

    # Analyze feature correlations
    analyze_correlations(df, OUTPUT_DIR)
