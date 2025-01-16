import torch

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\scaled_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\scaled_labels.pt"

# Load Data and Labels
data = torch.load(DATA_PATH).numpy()
labels = torch.load(LABELS_PATH).numpy()

# Print Shapes
print(f"Data Shape: {data.shape}")
print(f"Labels Shape: {labels.shape}")

# Check for mismatch
if data.shape[0] != labels.shape[0]:
    print("Mismatch detected. Resolving...")
    min_samples = min(data.shape[0], labels.shape[0])
    data = data[:min_samples]
    labels = labels[:min_samples]
    print(f"Resolved Data Shape: {data.shape}")
    print(f"Resolved Labels Shape: {labels.shape}")

    # Save fixed data and labels
    OUTPUT_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB"
    torch.save(torch.tensor(data, dtype=torch.float32), f"{OUTPUT_DIR}\\fixed_data.pt")
    torch.save(torch.tensor(labels, dtype=torch.long), f"{OUTPUT_DIR}\\fixed_labels.pt")
