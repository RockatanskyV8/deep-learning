import pandas as pd
import numpy as np

from Reader import *
from Plots  import *
from Modelos import *

from Runner import Regressor
from Treinamento import Treinamento

def plots(model, train_loss, valid_loss, X_test, y_test):
  print(model)
  p = Plots()
  p.plot_losses(train_loss, valid_loss)

  accuracy_final = p.get_accuracy(model, X_test, y_test)
  print("\n############ Acurácia ############")
  print(accuracy_final)
  print("############ -------- ############")

def run_models(dataset, criterion, layers, 
               run_epochs = 2000, run_early_stopping_epochs = 60, 
               input_features = 3, learning_rates = [], 
               run_retries = 5, run_batch_sizes = []):
  
  runner = Regressor(dataset, 
                     epochs = run_epochs, 
                     batch_sizes = run_batch_sizes,
                     early_stopping_epochs = run_early_stopping_epochs, 
                     retries = run_retries )
  
  return runner.regressao(input_features, layers, learning_rates, criterion)

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

random_seed = 42

input_features = 3
output_size = 2

df = pd.read_csv('../df_points.txt', sep='\t', index_col=[0])

p = 0.5
camadas = []
camadas.append(Camada(500, nn.ReLU(), nn.Dropout(p) ))
camadas.append(Camada(250, nn.ReLU(), nn.Dropout(p) ))
camadas.append(Camada(100, nn.ReLU(), nn.Dropout(p) ))
camadas.append(Camada( 10, nn.ReLU()))
camadas.append(Camada(output_size, nn.Softmax(dim=-1) ))

criterion = nn.CrossEntropyLoss()

current_valid_loss = run_models(df, criterion, camadas,
                                run_epochs = 2000, 
                                run_early_stopping_epochs = 75, 
                                input_features = 3,  
                                learning_rates = [0.0001, 0.001, 0.01, 0.1],  
                                run_retries = 5,
                                run_batch_sizes = [15, 30, 60, 120])


