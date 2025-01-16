import torch
from sklearn.utils import resample
import numpy as np

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

def balance_dataset(data, labels):
    """
    Balances the dataset by resampling classes to have equal representation.

    Args:
        data (torch.Tensor): Input data tensor.
        labels (torch.Tensor): Labels tensor (must be 1D).

    Returns:
        torch.Tensor, torch.Tensor: Balanced data and labels tensors.
    """
    labels = labels.view(-1)
    unique_labels, counts = torch.unique(labels, return_counts=True)
    max_count = max(counts).item()

    balanced_data = []
    balanced_labels = []

    for label in unique_labels:
        class_data = data[labels == label]
        class_labels = labels[labels == label]
        resampled_data, resampled_labels = resample(
            class_data.numpy(), class_labels.numpy(),
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        balanced_data.append(torch.tensor(resampled_data, dtype=torch.float32))
        balanced_labels.append(torch.tensor(resampled_labels, dtype=torch.long))

    return torch.cat(balanced_data), torch.cat(balanced_labels)

def balance_test_dataset(data, labels, num_samples_per_class):
    """
    Balances the test dataset by sampling an equal number of samples for each class.

    Args:
        data (torch.Tensor): Input test data.
        labels (torch.Tensor): Labels for the test data.
        num_samples_per_class (int): Desired number of samples per class.

    Returns:
        torch.Tensor, torch.Tensor: Balanced test data and labels.
    """
    labels = labels.view(-1)
    unique_labels = torch.unique(labels)
    balanced_data = []
    balanced_labels = []

    for label in unique_labels:
        class_data = data[labels == label]
        class_labels = labels[labels == label]

        if len(class_data) >= num_samples_per_class:
            indices = torch.randperm(len(class_data))[:num_samples_per_class]
            balanced_data.append(class_data[indices])
            balanced_labels.append(class_labels[indices])
        else:
            repeats = (num_samples_per_class + len(class_data) - 1) // len(class_data)
            class_data = class_data.repeat(repeats, 1)[:num_samples_per_class]
            class_labels = class_labels.repeat(repeats)[:num_samples_per_class]
            balanced_data.append(class_data)
            balanced_labels.append(class_labels)

    return torch.cat(balanced_data), torch.cat(balanced_labels)

def augment_data(data, labels):
    """
    Applies diverse augmentations to the input data and adjusts labels.

    Args:
        data (torch.Tensor): Input data tensor.
        labels (torch.Tensor): Corresponding labels tensor.

    Returns:
        torch.Tensor, torch.Tensor: Augmented data and labels.
    """
    augmented_data = []
    augmented_labels = []

    for signal, label in zip(data, labels):
        augmented_data.append(signal.float().unsqueeze(0))
        augmented_labels.append(label)

        noise = torch.normal(0, 0.05, size=signal.shape).float()
        augmented_data.append((signal + noise).unsqueeze(0))
        augmented_labels.append(label)

        scale = torch.rand(1) * 0.4 + 0.8
        augmented_data.append((signal * scale).unsqueeze(0))
        augmented_labels.append(label)

        time_steps = np.arange(len(signal))
        warp = np.interp(time_steps + np.random.normal(0, 1, size=len(time_steps)), time_steps, signal.numpy())
        augmented_data.append(torch.tensor(warp).float().unsqueeze(0))
        augmented_labels.append(label)

        fft_signal = np.fft.fft(signal.numpy())
        fft_signal += np.random.normal(0, 0.1, size=fft_signal.shape)
        noisy_signal = np.fft.ifft(fft_signal).real
        augmented_data.append(torch.tensor(noisy_signal).float().unsqueeze(0))
        augmented_labels.append(label)

    return torch.cat(augmented_data), torch.tensor(augmented_labels, dtype=torch.long)

def split_and_save_dataset(data, labels, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Splits the dataset into train, validation, and test sets and saves them.

    Args:
        data (torch.Tensor): Augmented dataset.
        labels (torch.Tensor): Corresponding labels.
        output_dir (str): Directory to save the splits.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for testing set.
    """
    from sklearn.model_selection import train_test_split

    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=(1 - train_ratio), stratify=labels, random_state=42
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=(test_ratio / (val_ratio + test_ratio)), stratify=temp_labels, random_state=42
    )

    torch.save(train_data, f"{output_dir}/train.pt")
    torch.save(train_labels, f"{output_dir}/train_labels.pt")
    torch.save(val_data, f"{output_dir}/val.pt")
    torch.save(val_labels, f"{output_dir}/val_labels.pt")
    torch.save(test_data, f"{output_dir}/test.pt")
    torch.save(test_labels, f"{output_dir}/test_labels.pt")

    print(f"Data splits saved to {output_dir}.")

    print(f"Train Data Size: {train_data.shape}")
    print(f"Validation Data Size: {val_data.shape}")
    print(f"Test Data Size: {test_data.shape}")

