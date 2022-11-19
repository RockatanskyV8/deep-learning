import torch
import torch.nn as nn

from datetime import datetime

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Runner import *
from Modelos import *
from Treinamento import Treinamento
################################################################################################

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

X_train.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)


input_features = 3
ss = StandardScaler()
ss.fit(X_train)

X_train = ss.transform(X_train)
X_valid = ss.transform(X_valid)
X_test = ss.transform(X_test)

y_train = pd.get_dummies(y_train, prefix='target').reset_index(drop=True)
y_valid = pd.get_dummies(y_valid, prefix='target').reset_index(drop=True)
y_test = pd.get_dummies(y_test, prefix='target').reset_index(drop=True)

model = MinhaNovaRede
#  optimizer = optim.SGD(model.parameters(), lr=0.01)

#  print(model)

criterion = nn.CrossEntropyLoss()

epochs = 2000
early_stopping_epochs = 20
best_valid_loss = np.Inf

learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [15, 30, 60, 120]
# number_of_hidden_layers = []
# neurons_in_hidden_layers = []


r = Runner(model, epochs, 0, early_stopping_epochs, retries = 5, batches = batch_sizes)
current_valid_loss = r.regressao(input_features, learning_rates, criterion, 
                                 torch.from_numpy(X_train),
                                 torch.from_numpy(y_train.to_numpy()),
                                 torch.from_numpy(X_valid),
                                 torch.from_numpy(y_valid.to_numpy()))

p = Plots()
accuracy_final = p.get_loss(model,
           torch.from_numpy(X_test.to_numpy()),
           torch.from_numpy(y_test),
           criterion)
print(accuracy_final)
