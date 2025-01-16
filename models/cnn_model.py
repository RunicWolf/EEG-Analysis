import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths to the data and labels
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_labels.pt"

# Load data function
def load_data(data_path, labels_path):
    data = torch.load(data_path).numpy()
    labels = torch.load(labels_path).numpy()
    return data, labels

# Label Encoding Function
def encode_labels(y):
    if len(y.shape) == 2:  # If labels are one-hot encoded
        y = np.argmax(y, axis=1)  # Convert to integers
    y = np.clip(y, 0, 3)  # Ensure labels are in the range 0-3
    return y.astype(int)

# Define CNN Model for EEG Classification
class EEG_CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Calculate the flattened size after convolution and pooling
        self.flattened_size = self._get_conv_output_size(input_channels)

        self.fc1 = nn.Linear(self.flattened_size, 256)  # Adjusted dynamically
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def _get_conv_output_size(self, input_channels):
        # Calculate the output size after conv1, pool, conv2, pool
        x = torch.zeros(1, input_channels, 420)  # (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x.numel()  # Return the total number of elements after pooling
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Dynamically calculate the flattened size
        x = torch.flatten(x, 1)  # Flatten the tensor except the batch dimension
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess data
X, y = load_data(DATA_PATH, LABELS_PATH)

# Encode the labels
y = encode_labels(y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN input (batch_size, channels, seq_length)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # (num_samples, 1, num_features)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])      # (num_samples, 1, num_features)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Initialize the model
model = EEG_CNN(input_channels=1, num_classes=4)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)  # Compute loss

    # Backward pass
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "cnn_model.pth")
print("CNN Model trained and saved.")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

    correct = (predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
