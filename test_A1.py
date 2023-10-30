import tqdm
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from common_utils import set_seed

# setting seed
set_seed()

class MLP(nn.Module):

    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            # YOUR CODE HERE
            #hidden layer1
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            #hidden layer2
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            #hidden layer3
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            #output layer
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    # YOUR CODE HERE
    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits


from common_utils import split_dataset, preprocess_dataset


def preprocess(df):
    # YOUR CODE HERE
    X_train_scaled, y_train, X_test_scaled, y_test = split_dataset(df, columns_to_drop=["filename", "label"],
                                                                   test_size=0.3, random_state=0)
    X_train_scaled, X_test_scaled = preprocess_dataset(X_train_scaled, X_test_scaled)

    return X_train_scaled, y_train, X_test_scaled, y_test


df = pd.read_csv('simplified.csv')
df['label'] = df['filename'].str.split('_').str[-2]

df['label'].value_counts()

X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df)


class CustomDataset(Dataset):
    # YOUR CODE HERE
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        label = self.y[idx]
        return x, label


def intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test):
    # YOUR CODE HERE
    train_dataset = CustomDataset(X_train_scaled, y_train)
    test_dataset = CustomDataset(X_test_scaled, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    return train_dataloader, test_dataloader


train_dataloader, test_dataloader = intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test)

# YOUR CODE HERE
# Hyperparameters
drop_out = 0.2
no_features = 77
no_hidden = 128
no_labels = 1
lr = 0.001

model = MLP(no_features=no_features, no_hidden=no_hidden, no_labels=no_labels) # 77 from assignment note, label = positive or negative emotions
                                                        # (aim is to determine the speech polarity of the engineered
                                                        # feature dataset)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# YOUR CODE HERE
def train_loop(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct += (pred.round() == y).type(torch.float).sum().item()
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss, correct


def test_loop(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (pred.round() == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

# Hyperparameters
patience = 3
num_epochs = 100
learning_rate = 0.001


from common_utils import EarlyStopper
early_stopper = EarlyStopper(patience=patience, min_delta=0)

train_loss, test_loss = [], []
train_acc, test_acc = [], []
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    train_loss.append(loss), train_acc.append(acc)

    loss, acc = test_loop(test_dataloader, model, loss_fn)
    test_loss.append(loss), test_acc.append(acc)


    if early_stopper.early_stop(loss):
        print("Early stop detected!")
        break

print("Done!")
