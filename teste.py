import torch
import torch.nn as nn
import torch.optim as optim

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Runner import *
from Modelos import *
from Reader import *
from Treinamento import Treinamento
################################################################################################
input_features = 3
df = pd.read_csv('../df_points.txt', sep='\t', index_col=[0])

r = Reader(df)
train, valid = r.read()

X_train, X_test, y_train, y_test = train
X_valid, X_test, y_valid, y_test = valid

model = MinhaNovaRede(input_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model)

epochs = 2000
batch_size = 25
early_stopping_epochs = 50 # quantas épocas sem melhoria serão toleradas antes de parar o treinamento

r = Runner(model, epochs, batch_size, early_stopping_epochs)
model, train_loss, valid_loss = r.classificacao(optimizer, criterion,
                                                torch.from_numpy(X_train),
                                                torch.from_numpy(y_train.to_numpy()),
                                                torch.from_numpy(X_valid),
                                                torch.from_numpy(y_valid.to_numpy()))

p = Plots()
accuracy_final = p.get_accuracy(model, torch.from_numpy(X_test), torch.from_numpy(y_test.to_numpy()))

print(accuracy_final)

