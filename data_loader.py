import torch

def load_data(file_path):
    """
    Loads data from a .pt file.

    Args:
        file_path (str): Path to the .pt file.

    Returns:
        np.ndarray: Loaded data.
    """
    return torch.load(file_path).cpu().numpy()

def save_data(data, file_path):
    """
    Saves data to a .pt file.

    Args:
        data (np.ndarray): Data to save.
        file_path (str): Path to save the .pt file.
    """
    torch.save(torch.tensor(data, dtype=torch.float32), file_path)
    print(f"Data saved to {file_path}")
