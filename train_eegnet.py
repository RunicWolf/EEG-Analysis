import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from data_loader import balance_test_dataset, augment_data, balance_dataset

class EEGNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EEGNet, self).__init__()
        self.temporal_conv = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.depthwise_conv = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, groups=16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.pointwise_conv = nn.Conv1d(32, 32, kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.7)
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.pointwise_conv(x)

        if self.fc is None:
            flatten_dim = x.size(1) * x.size(2)
            self.fc = nn.Linear(flatten_dim, self.num_classes).to(x.device)

        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=30):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.tolist())
                val_labels.extend(y_val.tolist())
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_acc:.4f}")

train_data_path = "train.pt"
train_labels_path = "train_labels.pt"

data, labels = torch.load(train_data_path).float(), torch.load(train_labels_path).long()

train_data, temp_data, train_labels, temp_labels = train_test_split(
    data, labels, test_size=0.4, stratify=labels, random_state=42
)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_data, train_labels = balance_dataset(train_data, train_labels)
train_data, train_labels = augment_data(train_data, train_labels)

train_data = train_data.unsqueeze(1).float()
val_data = val_data.unsqueeze(1).float()
test_data = test_data.unsqueeze(1).float()

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = EEGNet(input_dim=4, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=30)

model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        _, preds = torch.max(outputs, dim=1)
        test_preds.extend(preds.tolist())
        test_true.extend(y_test.tolist())

print(f"Test Accuracy: {accuracy_score(test_true, test_preds):.4f}")
print("Classification Report:")
print(classification_report(test_true, test_preds))
