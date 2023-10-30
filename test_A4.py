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
from common_utils import MLP, CustomDataset, generate_cv_folds, train_loop, test_loop, split_dataset, EarlyStopper, preprocess_dataset

# YOUR CODE HERE
import shap

df = 0
size_row = 1
size_column = 78
# YOUR CODE HERE

df = pd.read_csv('new_record.csv')
df

################################
################################
################################

def preprocess(X_train, df):
    """preprocess your dataset to obtain your test dataset, remember to remove the 'filename' as Q1
    """
    # YOUR CODE HERE
    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled_eg = standard_scaler.transform(df)

    return X_test_scaled_eg


test_df = df.drop(columns=['filename'])

df = pd.read_csv('simplified.csv')
# df['label'] = df['filename'].str.split('_').str[-2]
# df['label'].value_counts()
X_train = df.drop(columns=['filename'])

# X_train, y_train, X_test, y_test = split_dataset(df, ['filename','label'], test_size=0.3, random_state=0)
X_test_scaled_eg = preprocess(X_train, test_df)

# YOUR CODE HERE
# Load the optimized pretrained model
model = MLP(no_features=77, no_hidden=256, no_labels=1)  # Load the model with the optimal number of neurons
model.load_state_dict(torch.load('model_weights.pth'))  # Load the best model weights

# Set the model to evaluation mode
model.eval()

# Make predictions on the test dataset
with torch.no_grad():
    pred = model(torch.tensor(X_test_scaled_eg, dtype=torch.float))

# Apply the threshold of 0.5 to get the predicted label
threshold = 0.5
pred_label = 1 if (pred >= threshold) else 0

# Print the predicted labels
print("pred:", pred)
print(pred_label)

'''
Fit the explainer on a subset of the data (you can try all but then gets slower)
Return approximate SHAP values for the model applied to the data given by X.
Plot the local feature importance with a force plot and explain your observations.
'''
# YOUR CODE HERE
standard_scaler = preprocessing.StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train)
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
# Step 2: Create a DeepExplainer instance
explainer = shap.DeepExplainer(model, X_train_scaled_tensor)

# Step 3: Select a specific test sample
# Calculate the number of samples to take (10% of the total)
subset_size = int(0.001 * X_train_scaled_tensor.shape[0])

# Randomly select a 1% subset of the data
subset_indices = np.random.choice(X_train_scaled_tensor.shape[0], subset_size, replace=False)
subset_test_sample = X_train_scaled_tensor[subset_indices]

# Step 4: Compute SHAP values for the test sample
shap_values = explainer.shap_values(subset_test_sample, ranked_outputs=1)

# Step 5: Create a force plot
shap.initjs()  # Initialize the JavaScript visualization library
shap.force_plot(explainer.expected_value[0], shap_values[0], subset_test_sample, matplotlib=True, show=False)

# Step 6: Display and interpret the force plot
plt.show()
