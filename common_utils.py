### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

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


def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])
    # print(df[['label','filename']])
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop, axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop, axis=1)
    y_test2 = df_test['label'].to_numpy()

    return df_train2, y_train2, df_test2, y_test2


def preprocess_dataset(df_train, df_test):
    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled


def set_seed(seed=0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# My own code below
class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) model.
    """
    def __init__(self, no_features, no_hidden, no_labels, drop_out=0.2):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            # YOUR CODE HERE
            # hidden layer1
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            # hidden layer2
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            # hidden layer3
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            # output layer
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    # YOUR CODE HERE
    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits


class CustomDataset(Dataset):
    """
    Custom dataset for PyTorch DataLoader.
    """
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


# YOUR CODE HERE
def train_loop(train_dataloader, model, loss_fn, optimizer):
    """
    Training loop for the model.

    Args:
        train_dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        loss_fn: The loss function.
        optimizer: The optimizer for training.

    Returns:
        train_loss (float): Average training loss.
        correct (float): Training accuracy.
    """

    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        correct += (pred.round() == y).type(torch.float).sum().item()
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    # correct /= size
    correct = float(correct) / size
    print(f"Train Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss, correct


def test_loop(test_dataloader, model, loss_fn):
    """
    Testing loop for the model.

    Args:
        test_dataloader (DataLoader): DataLoader for testing data.
        model (nn.Module): The neural network model.
        loss_fn: The loss function.

    Returns:
        test_loss (float): Average testing loss.
        correct (float): Testing accuracy.
    """
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct = float(correct) / size
    print(f"Test Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


# Used in PartA_3
def generate_cv_folds(parameters, X_train, y_train):
    """
    Generate cross-validation folds for hyperparameter optimization. This is different from the original implementation.

    Args:
        parameters (dict): Hyperparameters for cross-validation.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.

    Returns:
        X_train_scaled_dict (dict): Training data for different folds.
        X_val_scaled_dict (dict): Validation data for different folds.
        y_train_dict (dict): Training labels for different folds.
        y_val_dict (dict): Validation labels for different folds.
    """
    # YOUR CODE HERE

    X_train_scaled_dict = {}  # Store preprocessed training data for different batch sizes
    X_val_scaled_dict = {}    # Store preprocessed validation data for different batch sizes
    y_train_dict = {}         # Store labels for training data for different batch sizes
    y_val_dict = {}           # Store labels for validation data for different batch sizes

    for num_neurons in parameters["num_neurons"]:
        kfold = KFold(n_splits=parameters["cv_fold"], shuffle=True, random_state=0)

        X_train_scaled_folds = []  # Store preprocessed training data for each fold
        X_val_scaled_folds = []    # Store preprocessed validation data for each fold
        y_train_folds = []         # Store labels for training data for each fold
        y_val_folds = []           # Store labels for validation data for each fold

        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Scale data
            X_train_fold_scaled, X_val_fold_scaled = preprocess_dataset(X_train_fold, X_val_fold)

            X_train_scaled_folds.append(X_train_fold_scaled)
            X_val_scaled_folds.append(X_val_fold_scaled)
            y_train_folds.append(y_train_fold)
            y_val_folds.append(y_val_fold)

        # Store data for the current batch size
        X_train_scaled_dict[num_neurons] = X_train_scaled_folds
        X_val_scaled_dict[num_neurons] = X_val_scaled_folds
        y_train_dict[num_neurons] = y_train_folds
        y_val_dict[num_neurons] = y_val_folds

    return X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict

