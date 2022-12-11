import pandas as pd
import numpy as np

from Reader import *
from Plots  import *
from Modelos import *

from Treinamento import Treinamento
from datetime import datetime

################################################################################################

class Regressor():
    def __init__(self, dataset, epochs = 2000, batch_sizes = [], early_stopping_epochs = 60, retries = 5):
        self.dataset               = dataset
        self.epochs                = epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.batch_sizes           = batch_sizes
        self.retries               = retries

    def regressao(self, input_features, layers, learning_rates, criterion):
        start = datetime.now()
        current_valid_loss = 0
        best_valid_loss = np.Inf
        train_loss, valid_loss = [], []

        for initializations in range(0, self.retries):
            for learning_rate in learning_rates:
                for batch_size in self.batch_sizes:

                    print(f'----------\nLearning rate: {learning_rate}\nBatch size: {batch_size}\n')

                    model = GeradorRede(input_features, layers)
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                    t = Treinamento(self.dataset, self.epochs, batch_size, self.early_stopping_epochs)
                    model, train_loss, valid_loss = t.train_linear(model, optimizer, criterion)

                    # store best valid loss
                    current_valid_loss = min(valid_loss)
                    if current_valid_loss < best_valid_loss:
                        torch.save(model.state_dict(), 'best_global_model')
                        best_valid_loss = current_valid_loss
                        print('New best global model found!')

                    print(f'\nValidation loss: {current_valid_loss}\n')
        
        end = datetime.now()
        print(f'\n\n\n--------------------\nTotal training time: {end - start}')

################################################################################################

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


