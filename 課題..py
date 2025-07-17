import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/yukishimomura/Downloads/drug200.csv')

X = df.drop('Drug', axis=1)
y = df['Drug']

# カテゴリカルな特徴量を数値に変換
categorical_cols = ['Sex', 'BP', 'Cholesterol']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ターゲット変数（薬の種類）を数値ラベルに変換
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 数値特徴量を標準化
scaler = StandardScaler()
numerical_cols = ['Age', 'Na_to_K']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

class DrugClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(DrugClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_features, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x

input_size = X_train.shape[1]
output_size = len(label_encoder.classes_)
model = DrugClassifier(input_features=input_size, num_classes=output_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
loss_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'\nTest Accuracy: {accuracy:.2f} %')

