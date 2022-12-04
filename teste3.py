import pandas as pd
import numpy as np

from Plots  import *
from Modelos import *

from Runner import Regressor
from Treinamento import Treinamento

from teste_CrossValidation import Treinamento
from Reader import *

random_seed = 42
input_features = 3
df = pd.read_csv('../df_points.txt', sep='\t', index_col=[0])

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

input_features = 3
output_size = 2
p = 0.5
camadas = []
camadas.append(Camada( 250, nn.ReLU(), nn.Dropout(p) ))
camadas.append(Camada( 100, nn.ReLU(), nn.Dropout(p) ))
camadas.append(Camada(  10, nn.ReLU()))
camadas.append(Camada(output_size, nn.Softmax(dim=-1) ))

model = GeradorRede(input_features , camadas)

#  print(model)
#  print("\n ----------------------------------- ")
read = Reader(df)
train, valid = read.read()

X_train, X_test, y_train, y_test = train
X_valid, X_test, y_valid, y_test = valid

t = Treinamento()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

t.train(model, 
        10, 20, 60, 
        optimizer, criterion, df)

