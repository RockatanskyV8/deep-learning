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
df = pd.read_csv('../df_points.txt', sep='\t', index_col=[0])

r = Reader(df)
train, valid = r.read()

X_train, X_test, y_train, y_test = train
X_valid, X_test, y_valid, y_test = valid

softmax = nn.Softmax(dim=-1)

input_features = 3
output_size = 2
p = 0.5

camada1 = Camada(500, nn.ReLU(), nn.Dropout(p) )
camada2 = Camada(500, nn.ReLU(), nn.Dropout(p) )
camada3 = Camada(2, softmax)

model = GeradorRede( input_features , [Camada(500, nn.ReLU(), nn.Dropout(p) ),
                                       Camada(250, nn.ReLU(), nn.Dropout(p) ),
                                       Camada(100, nn.ReLU(), nn.Dropout(p) ),
                                       Camada( 10, nn.ReLU(), nn.Dropout(p) ),
                                       Camada(  2, softmax ),
                                       ])
#  model = MinhaNovaRede(input_features)
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

#  (camada_entrada): Linear(in_features=3, out_features=500, bias=True)
#  (camada_oculta_1): Linear(in_features=500, out_features=250, bias=True)
#  (camada_oculta_2): Linear(in_features=250, out_features=100, bias=True)
#  (camada_oculta_3): Linear(in_features=100, out_features=10, bias=True)
#  (camada_saida): Linear(in_features=10, out_features=2, bias=True)
#  (relu): ReLU()
#  (softmax): Softmax(dim=-1)
#  (dropout_1): Dropout(p=0.125, inplace=False)
#  (dropout_2): Dropout(p=0.25, inplace=False)
#  (dropout_3): Dropout(p=0.375, inplace=False)

