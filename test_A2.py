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
from common_utils import MLP, CustomDataset, preprocess_dataset, train_loop, test_loop

print("torch.cuda.is_available: ", torch.cuda.is_available())
print("torch.cuda.device_count ", torch.cuda.device_count())
print("torch.cuda.current_device ", torch.cuda.current_device())
print("torch.cuda.device ", torch.cuda.device(0))
print("torch.cuda.get_device_name ", torch.cuda.get_device_name(0))


def generate_cv_folds_for_batch_sizes(parameters, X_train, y_train):
    """
    returns:
    X_train_scaled_dict(dict) where X_train_scaled_dict[batch_size] is a list of the preprocessed training matrix for the different folds.
    X_val_scaled_dict(dict) where X_val_scaled_dict[batch_size] is a list of the processed validation matrix for the different folds.
    y_train_dict(dict) where y_train_dict[batch_size] is a list of labels for the different folds
    y_val_dict(dict) where y_val_dict[batch_size] is a list of labels for the different folds
    """
    # YOUR CODE HERE
    X_train_scaled_dict = {}  # Store preprocessed training data for different batch sizes
    X_val_scaled_dict = {}  # Store preprocessed validation data for different batch sizes
    y_train_dict = {}  # Store labels for training data for different batch sizes
    y_val_dict = {}  # Store labels for validation data for different batch sizes

    for batch_size in parameters:
        kfold = KFold(n_splits=cv_fold, shuffle=True, random_state=0)

        X_train_scaled_folds = []  # Store preprocessed training data for each fold
        X_val_scaled_folds = []  # Store preprocessed validation data for each fold
        y_train_folds = []  # Store labels for training data for each fold
        y_val_folds = []  # Store labels for validation data for each fold

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
        X_train_scaled_dict[batch_size] = X_train_scaled_folds
        X_val_scaled_dict[batch_size] = X_val_scaled_folds
        y_train_dict[batch_size] = y_train_folds
        y_val_dict[batch_size] = y_val_folds
    return X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict


# Prepare dataset
df = pd.read_csv('simplified.csv')
df['label'] = df['filename'].str.split('_').str[-2]
df['label'].value_counts()

label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
X_train = df.drop(["filename", "label"], axis=1)
y_train = df['label'].to_numpy()

batch_sizes = [128, 256, 512, 1024]
cv_fold = 5  # needed for function
X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict = generate_cv_folds_for_batch_sizes(batch_sizes,
                                                                                                     X_train.to_numpy(),
                                                                                                     y_train)

# YOUR CODE HERE
from statistics import mean

# Hyperparameters
drop_out = 0.2
no_features = 77
no_hidden = 128
no_labels = 1
lr = 0.001
num_epochs = 100


def find_optimal_hyperparameter(X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict, batch_sizes,
                                hyperparam):
    cross_validation_accuracies = {}
    cross_validation_times = {}

    for batch_size in batch_sizes:
        # Get test and train data for each batch size
        x_train = X_train_scaled_dict[batch_size]
        x_val = X_val_scaled_dict[batch_size]
        y_train = y_train_dict[batch_size]
        y_val = y_val_dict[batch_size]

        fold_accuracies = []
        timing = []

        for fold in range(len(x_train)):
            model = MLP(no_features=no_features, no_hidden=no_hidden, no_labels=no_labels)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.BCELoss()

            train_dataset = CustomDataset(x_train[fold], y_train[fold])
            val_dataset = CustomDataset(x_val[fold], y_val[fold])

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            for epoch in range(num_epochs):
                if epoch == 99:
                    start_time = time.time()
                print(f"Epoch {epoch + 1}\n--------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)

                if epoch == 99:
                    end_time = time.time()
                    timing.append(end_time - start_time)

            loss, accuracy = test_loop(val_dataloader, model, loss_fn)
            fold_accuracies.append(accuracy)

        # mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        # mean_time = mean(timing)
        mean_accuracy = np.mean(np.array(fold_accuracies))
        mean_time = np.mean(np.array(timing))

        cross_validation_accuracies[batch_size] = mean_accuracy
        cross_validation_times[batch_size] = mean_time

    return cross_validation_accuracies, cross_validation_times


batch_sizes = [128, 256, 512, 1024]
cross_validation_accuracies, cross_validation_times = find_optimal_hyperparameter(X_train_scaled_dict,
                                                                                  X_val_scaled_dict, y_train_dict,
                                                                                  y_val_dict, batch_sizes, 'batch_size')

# YOUR CODE HERE
mean_accuracies = []
for batch_size, mean_accuracy in cross_validation_accuracies.items():
    print(f"Mean Cross-Validation Accuracy for Batch Size {batch_size}: {mean_accuracy:.4f}")
    mean_accuracies.append(mean_accuracy)

# Create a scatterplot
plt.figure(1)
plt.scatter(batch_sizes, mean_accuracies, label='Mean Cross-Validation Accuracy')
plt.title('Mean Cross-Validation Accuracies for Different Batch Sizes')
plt.xlabel('Batch Size')
plt.xticks(batch_sizes)
plt.ylabel('Mean Accuracy')
plt.grid(True)

# Show the plot
plt.show()

last_epoch_times = {}

# Populate the dictionary with the last epoch times for each batch size
for batch_size, mean_time in cross_validation_times.items():
    last_epoch_times[batch_size] = mean_time

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(last_epoch_times.items()), columns=['Batch Size', 'Last Epoch Time'])

# YOUR CODE HERE
optimal_batch_size = 128
reason = f"""
Given sufficient computation power and time, I would select {optimal_batch_size} because it has the highest accuracy.
Comparison:
Mean Cross-Validation Accuracy for Batch Size 128: 0.7977
Mean Cross-Validation Accuracy for Batch Size 256: 0.7922
Mean Cross-Validation Accuracy for Batch Size 512: 0.7911
Mean Cross-Validation Accuracy for Batch Size 1024: 0.7839
"""



# Display the DataFrame
print(df)

# Display the optimal batch size and reason
print(f"Optimal Batch Size: {optimal_batch_size}")
print(f"Reason: {reason}")