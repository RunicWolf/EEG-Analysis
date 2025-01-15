import numpy as np

def augment_and_label(data, labels, num_classes=3, augment_factor=2):
    """
    Balances the dataset by creating synthetic labels and augmenting data.

    Args:
        data (np.ndarray): Original EEG data.
        labels (np.ndarray): Corresponding labels (single class).
        num_classes (int): Total number of synthetic classes to create.
        augment_factor (int): Number of augmentations per original sample.

    Returns:
        tuple: Augmented data and balanced labels.
    """
    augmented_data = []
    augmented_labels = []

    class_size = len(data) // num_classes

    for class_id in range(num_classes):
        start_idx = class_id * class_size
        end_idx = start_idx + class_size

        class_data = data[start_idx:end_idx]
        class_label = class_id

        for signal in class_data:
            augmented_data.append(signal)
            augmented_labels.append(class_label)

            for _ in range(augment_factor):
                # Gaussian noise
                noisy_signal = signal + np.random.normal(0, 0.01, size=signal.shape)

                # Amplitude scaling
                scaling_factor = np.random.uniform(0.8, 1.2)
                scaled_signal = signal * scaling_factor

                augmented_data.extend([noisy_signal, scaled_signal])
                augmented_labels.extend([class_label] * 2)

    return np.array(augmented_data), np.array(augmented_labels)
