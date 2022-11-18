########### my imports
import torch
import torch.nn as nn
import torch.optim as optim

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Modelos import *
from Treinamento import Treinamento
from Plots import Plots
################################################################################################
# other imports
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
################################################################################################

from torch.utils.data import Dataset

def reset_weights(m):
  # Try resetting model weights to avoid weight leakage.
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

#  if __name__ == '__main__':

random_seed = 42
pd.set_option('display.max_columns', None)

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

df = pd.read_csv('../df_points.txt', sep='\t', index_col=[0])

train_p = 0.7
val_p = 0.15
test_p = 0.15

train_size = int(train_p*df.shape[0])
val_size = int(val_p*df.shape[0])
test_size = int(test_p*df.shape[0])

X_train, X_test, y_train, y_test = train_test_split(df[['x', 'y', 'z']],
                                                    df['label'],
                                                    train_size=train_size,
                                                    stratify=df['label'],
                                                    random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(X_test,
                                                    y_test,
                                                    test_size=test_size,
                                                    stratify=y_test,
                                                    random_state=42)


input_features = 3
ss = StandardScaler()
ss.fit(X_train)

X_train = ss.transform(X_train)
X_valid = ss.transform(X_valid)
X_test = ss.transform(X_test)

y_train = pd.get_dummies(y_train, prefix='target').reset_index(drop=True)
y_valid = pd.get_dummies(y_valid, prefix='target').reset_index(drop=True)
y_test = pd.get_dummies(y_test, prefix='target').reset_index(drop=True)

num_epochs = 2000
batch_size = 25
early_stopping_epochs = 50 # quantas épocas sem melhoria serão toleradas antes de parar o treinamento

X_train = torch.from_numpy(X_train),
y_train = torch.from_numpy(y_train.to_numpy())
X_valid = torch.from_numpy(X_valid),
y_valid = torch.from_numpy(y_valid.to_numpy())

def get_batches(data, batch_size=1):
    batches = []
    
    data_size = len(data)
    for start_idx in range(0, data_size, batch_size):
        end_idx = min(data_size, start_idx + batch_size)
        batches.append(data[start_idx:end_idx])
    
    return batches





for index, (original_data, original_target) in enumerate(zip(get_batches(X_train, batch_size),
                                                             get_batches(y_train, batch_size))):
    # Format data to tensor
    #  target = (original_target == 1).nonzero(as_tuple=True)[1]
    #  data = original_data #.float() # Esse '.float()' é necessário para arrumar o tipo do dado
    #  network.apply(reset_weights)
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(original_data)
    test_subsampler  = torch.utils.data.SubsetRandomSampler(original_target)

    trainloader = torch.utils.data.DataLoader(
                      df, 
                      batch_size=10, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      df,
                      batch_size=10, sampler=test_subsampler)


    network = MinhaNovaRede()
    network.apply(reset_weights)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)


    for epoch in range(0, num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, targets = data
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = network(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():
        
      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):
          print(i)

#          # Get inputs
#          inputs, targets = data
#
#          # Generate outputs
#          outputs = network(inputs)
#
#          # Set total and correct
#          _, predicted = torch.max(outputs.data, 1)
#          total += targets.size(0)
#          correct += (predicted == targets).sum().item()
#
#        # Print accuracy
#        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
#        print('--------------------------------')
#        results[fold] = 100.0 * (correct / total)
#
#  # Print fold results
#  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
#  print('--------------------------------')
#  sum = 0.0
#  for key, value in results.items():
#    print(f'Fold {key}: {value} %')
#    sum += value
#  print(f'Average: {sum/len(results.items())} %')
        
    
