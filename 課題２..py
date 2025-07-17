import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/yukishimomura/Library/CloudStorage/OneDrive-神戸大学【全学】/Sport car price.csv')

# --- Data Cleaning and Selection ---
# As requested, we select only the relevant columns
features = ['Engine Size (L)', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)']
target = 'Price (in USD)'
columns_to_use = features + [target]
df_selected = df[columns_to_use].copy()

# Convert columns with non-numeric characters (like commas) to numbers
for col in ['Engine Size (L)','Horsepower', 'Torque (lb-ft)', 'Price (in USD)']:
    # Replace commas and convert to numeric, forcing errors to become NaN (Not a Number)
    df_selected[col] = pd.to_numeric(df_selected[col].astype(str).str.replace(',', ''), errors='coerce')

# Drop any rows that have missing values after conversion
df_selected.dropna(inplace=True)

# Separate features (X) and target (y)
X = df_selected[features]
y = df_selected[[target]] # Keep y as a DataFrame

print("Data after cleaning:")
print(X.head())

# --- Scaling and Splitting ---

# Initialize scalers for features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

# Scale the features and target
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert arrays to PyTorch Tensors
X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.float32))

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# --- Model Architecture ---

class PricePredictor(nn.Module):
    def __init__(self, input_features):
        super(PricePredictor, self).__init__()
        self.layer_1 = nn.Linear(input_features, 128)
        self.layer_2 = nn.Linear(128, 64)
        # The output layer has 1 neuron for the price prediction
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x) # No activation function for regression output
        return x

# Instantiate the model
input_size = X_train.shape[1]
model = PricePredictor(input_features=input_size)
print("\nModel Architecture:")
print(model)

# --- Training the Model ---

# Loss function for regression: Mean Squared Error
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
loss_history = []

for epoch in range(epochs):
    model.train() # Set the model to training mode
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
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# --- Evaluation and Visualization ---

model.eval() # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for inference
    # Make predictions on the test set
    predicted_scaled = model(X_test_tensor)

    # Reverse the scaling to get prices in USD
    predicted_prices = y_scaler.inverse_transform(predicted_scaled.numpy())
    actual_prices = y_scaler.inverse_transform(y_test)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    print(f'\nMean Absolute Error on Test Data: ${mae:,.2f}')

# Plotting Actual vs. Predicted Prices
plt.figure(figsize=(10, 8))
plt.scatter(actual_prices, predicted_prices, alpha=0.6, edgecolors='k')
plt.title('Actual vs. Predicted Car Prices', fontsize=16)
plt.xlabel('Actual Price (in USD)', fontsize=12)
plt.ylabel('Predicted Price (in USD)', fontsize=12)

# Add a reference line (y=x) for perfect predictions
max_price = max(actual_prices.max(), predicted_prices.max())
plt.plot([0, max_price], [0, max_price], 'r--', lw=2, label='Perfect Prediction')
plt.legend()
plt.grid(True)
plt.show()

