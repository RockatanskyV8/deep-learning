import torch
import torch.nn as nn
import torch.optim as optim

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Runner import Validacao_Cruzada
from Modelos import *
from Plots import *

################################################################################################
df = pd.read_csv('../df_points.txt', sep='\t', index_col=[0])

input_features = 3
output_size = 2
p = 0.5

model = GeradorRede( input_features , [Camada(500, nn.ReLU(), nn.Dropout(p) ),
                                       Camada(250, nn.ReLU(), nn.Dropout(p) ),
                                       Camada(100, nn.ReLU(), nn.Dropout(p) ),
                                       Camada( 10, nn.ReLU()),
                                       Camada(  2, nn.Softmax(dim=-1) ),
                                       ])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model)

epochs = 4000
batch_size = 5
early_stopping_epochs = 400 # quantas épocas sem melhoria serão toleradas antes de parar o treinamento

r = Validacao_Cruzada(df, model, epochs, batch_size, early_stopping_epochs)
model, train_loss, valid_loss, train, valid = r.classificacao(optimizer, criterion)

X_train, X_test, y_train, y_test = train
X_valid, X_test, y_valid, y_test = valid

p = Plots()
accuracy_final = p.get_accuracy(model, torch.from_numpy(X_test), torch.from_numpy(y_test.to_numpy()))
print(accuracy_final)

