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

# YOUR CODE HERE
from common_utils import MLP, CustomDataset, generate_cv_folds, train_loop, test_loop, split_dataset, EarlyStopper


def train(model, X_train_scaled, y_train2, X_val_scaled, y_val2, batch_size):
    # YOUR CODE HERE
    train_dataset = CustomDataset(X_train_scaled, y_train2)
    test_dataset = CustomDataset(X_val_scaled, y_val2)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=parameters["patience"], min_delta=0)

    # train_accuracy_list = []
    # train_losses_list = []
    # test_accuracy_list = []
    # test_losses_list = []

    for epoch in range(parameters["num_epochs"]):

        start = time.time()
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_losses, train_accuracies = train_loop(train_dataloader, model, loss_fn, optimizer)
        end = time.time()
        # train_accuracy_list.append(train_accuracies)
        # train_losses_list.append(train_losses)
        # test_accuracy_list.append(test_accuracies)
        # test_losses_list.append(test_losses)

        test_losses, test_accuracies = test_loop(test_dataloader, model, loss_fn)

        if early_stopper.early_stop(test_losses):
            print("Early Stop!")
            times = end - start

            return train_accuracies, train_losses, test_accuracies, test_losses, times

    return train_accuracies, train_losses, test_accuracies, test_losses, times


def find_optimal_hyperparameter(X_train, y_train, parameters, mode, batch_size):
    # YOUR CODE HERE
    # Scale data and generate cross folds
    X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict = generate_cv_folds(parameters, X_train, y_train)

    cross_validation_accuracies = []
    cross_validation_times = []

    # Initialize an empty list to store fold accuracies for the current number of hidden neurons
    # fold_train_accuracies = []
    # fold_test_accuracies = []
    # Iterate over different numbers of hidden neurons
    for num_hidden in parameters["num_neurons"]:
        # Get test and train data for each batch size
        x_train_list = X_train_scaled_dict[num_hidden]
        x_val_list = X_val_scaled_dict[num_hidden]
        y_train_list = y_train_dict[num_hidden]
        y_val_list = y_val_dict[num_hidden]

        fold_times = []
        fold_accuracies = []

        for fold in range(parameters["cv_fold"]):
            x_train = x_train_list[fold]
            x_val = x_val_list[fold]
            y_train = y_train_list[fold]
            y_val = y_val_list[fold]
            # Create and train the model with the current number of hidden neurons
            model = MLP(no_features=parameters["no_features"], no_hidden=num_hidden, no_labels=parameters["no_labels"])

            train_accuracies, train_losses, test_accuracies, test_losses, times = train(model, x_train, y_train,
                                                                                        x_val, y_val, batch_size)

            fold_times.append(times)
            fold_accuracies.append(test_accuracies)

        # Calculate the mean of fold metrics and store them
        # fold_train_accuracies = np.array(fold_train_accuracies) + np.array(train_accuracy_list) if \
        #     len(fold_train_accuracies) != 0 else np.array(train_accuracy_list)
        # fold_test_accuracies  = np.array(fold_test_accuracies) + np.array(test_accuracy_list) if \
        #     len(fold_test_accuracies) != 0 else np.array(test_accuracy_list)
        # fold_times  = np.array(fold_times) + np.array(times_list) if \
        #     len(fold_times) != 0 else np.array(times_list)

        cross_validation_accuracies.append(fold_accuracies)
        cross_validation_times.append(fold_times)

    cross_validation_accuracies = np.mean(np.array(cross_validation_accuracies), axis=1)
    cross_validation_times = np.mean(np.array(cross_validation_times), axis=1)

    return cross_validation_accuracies, cross_validation_times

'''
optimal_bs = 0. Fill your optimal batch size in the following code.
'''
# YOUR CODE HERE
# Prepare dataset
df = pd.read_csv('simplified.csv')
df['label'] = df['filename'].str.split('_').str[-2]
df['label'].value_counts()
# print(df[['label','filename']])

X_train, y_train, X_test, y_test = split_dataset(df, ['filename','label'], test_size=0.3, random_state=0)
X_train = pd.concat([X_train, X_test])
y_train = np.concatenate((y_train, y_test), axis=0)

parameters = {"num_neurons":[64, 128, 256],
             "cv_fold": 5,
              "no_features": 77,
              "no_labels": 1,
              "learning_rate": 0.001,
              "batch_size": [128],
              "num_epochs": 100,
              "patience": 3}

num_neurons = [64, 128, 256]
cross_validation_accuracies, cross_validation_times = find_optimal_hyperparameter(X_train.to_numpy(),
                                                                                  y_train,
                                                                                  parameters,
                                                                                  'num_neurons', 128)


# YOUR CODE HERE
# # Extract data for desired numbers of neurons (64, 128, 256)
# num_neurons = parameters["num_neurons"]
# accuracies_64 = cross_validation_accuracies[64]['test']  # Replace with your data
# accuracies_128 = cross_validation_accuracies[128]['test']  # Replace with your data
# accuracies_256 = cross_validation_accuracies[256]['test']  # Replace with your data
#
# num_epochs = parameters["num_epochs"]
# epochs = list(range(1, num_epochs + 1))
#
# # Create a plot
# plt.figure(figsize=(10, 6))
#
# # Plot accuracies for each number of neurons
# plt.plot(epochs, accuracies_64, label='64 Neurons', marker='o', linestyle='-')
# plt.plot(epochs, accuracies_128, label='128 Neurons', marker='x', linestyle='-')
# plt.plot(epochs, accuracies_256, label='256 Neurons', marker='o', linestyle='-')
#
# plt.xlabel('Epochs')
# plt.ylabel('Cross-validation accuracy')
# plt.title('Cross-validation accuracies vs Number of Epochs')
# plt.legend()
#
# # Display the plot
# plt.show()

num_neurons = parameters["num_neurons"]
plt.figure(1)
plt.plot(num_neurons, cross_validation_accuracies, marker = 'x', linestyle = 'None')
plt.xticks(num_neurons)
plt.xlabel('num neurons')
plt.ylabel('cross-validation accuracy')
plt.show()


optimal_neurons = 256
reason = "Selected because it had the highest cross-validation accuracy on last epoch."
# YOUR CODE HERE
df = pd.DataFrame({'Number of neurons': num_neurons,
                   'Last Epoch Time': cross_validation_times
                  })

print("df:", df)

# YOUR CODE HERE
_train, X_test, y_test = split_dataset(df, ['filename', 'label'], test_size=0.3, random_state=0)
X_train, X_test = preprocess_dataset(X_train, X_test)

train_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = MLP(no_inputs, 256, no_hidden, no_outputs)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

early_stopper = EarlyStopper(patience=parameters["patience"], min_delta=0)

train_loss, test_losses = [], []
train_acc, test_acc = [], []
for t in range(no_epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    train_loss.append(loss), train_acc.append(acc)

    loss, acc = test_loop(test_dataloader, model, loss_fn)
    test_losses.append(loss), test_acc.append(acc)

    if early_stopper.early_stop(loss):
        print("Done!")
        break

print("Done!")