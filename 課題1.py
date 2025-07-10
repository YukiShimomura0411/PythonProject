import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pd.read_csv('/Users/yukishimomura/Library/CloudStorage/OneDrive-神戸大学【全学】/卒論データ.csv', index_col = None, header = 0, encoding = "cp932")

X = data.drop(["win_1"], axis=1)
T = data[["win_1"]]

X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.1, random_state = 42)

DataList = []
for i in range(66):
    DataList.append(f'PC{i+1}')

x_train = X_train[DataList]
t_train = T_train["win_1"]

x_test = X_test[DataList]
t_test = T_test["win_1"]

x_train= torch.tensor(x_train.values, dtype=torch.float32)
x_test= torch.tensor(x_test.values, dtype=torch.float32)
t_train= torch.tensor(t_train.values, dtype=torch.long)
t_test= torch.tensor(t_test.values, dtype=torch.long)

t_train = t_train.reshape((-1,1))
t_test = t_test.reshape((-1,1))

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(x_train, t_train)
test_dataset = TensorDataset(x_test, t_test)

batch_size = 430
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

import torch.nn.functional as F
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.model_info = nn.ModuleList([
             nn.Linear(66,100),
             nn.ReLU(),
             nn.BatchNorm1d(100),
             nn.Dropout(0.5),
             nn.Linear(100,1),
             nn.Sigmoid()
            ])

    def forward(self, x):
        for i in range(len(self.model_info)):
            x = self.model_info[i](x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model().to(device)

from torch import optim
criterion = nn.BCELoss()
l1_lambda = 0.0001
l2_lambda = 0.0001
optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay = l2_lambda)
for param in model.parameters():
    param.register_hook(lambda grad: l1_lambda * torch.sign(grad))
num_epochs = 100

loss_train_history = []
loss_test_history = []
accuracy_train_list = []
accuracy_test_list = []

for epoch in range(num_epochs):
  model.train()

  epoch_train_loss = 0
  epoch_test_loss = 0
  num_train_batches = 0
  num_test_batches = 0
  correct_train = 0
  correct_test = 0

  for x,t in train_dataloader:
    x = x.to(device)
    t = t.to(device).float()
    optimizer.zero_grad()
    y = model(x)
    loss_train = criterion(y,t)
    epoch_train_loss += loss_train.item()
    num_train_batches += 1
    pred_train = torch.where(y < 0.5,0,1)
    correct_train += pred_train.eq(t.view_as(pred_train)).sum().item()
    loss_train.backward()
    optimizer.step()

  model.eval()

  with torch.no_grad():
    for x,t in test_dataloader:
      x = x.to(device)
      t = t.to(device).float()
      y = model(x)
      loss_test = criterion(y,t)
      epoch_test_loss += loss_test.item()
      num_test_batches += 1
      pred_test = torch.where(y < 0.5,0,1)
      correct_test += pred_test.eq(t.view_as(pred_test)).sum().item()

  avg_train_loss = epoch_train_loss / num_train_batches
  loss_train_history.append(avg_train_loss)
  avg_test_loss = epoch_test_loss / num_test_batches
  loss_test_history.append(avg_test_loss)
  avg_train_acc = correct_train / len(train_dataset)
  accuracy_train_list.append(avg_train_acc)
  avg_test_acc = correct_test / len(test_dataset)
  accuracy_test_list.append(avg_test_acc)


  print(f"Epoch: {epoch+1}/{num_epochs}, Train_Loss: {avg_train_loss:.4f}, Train_Acc: {avg_train_acc:.4f}, Test Loss: {avg_test_loss:.4f}, Test_Acc: {avg_test_acc:.4f}")

