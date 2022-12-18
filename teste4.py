import torch
import torch.nn as nn
import torch.optim as optim

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import datetime

from Treinamento import Treinamento
from Modelos import *
from Plots import *


class Regressor():
    def __init__(self, dataset, epochs = 2000, batch_sizes = [], early_stopping_epochs = 60, retries = 1):
        self.dataset               = dataset
        self.epochs                = epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.batch_sizes           = batch_sizes
        self.retries               = retries

    def model_generator(self, input_feature, neuronios, layers, p = .5, output_size = 2):
        camadas = []
        for i in range(0, layers):
            drp = random.uniform(0, p)
            c = Camada( int(neuronios/2**i), nn.ReLU(), nn.Dropout(drp) )
            camadas.append(c)

        camadas.append(  Camada( output_size, nn.Softmax(dim=-1) ) )
        return GeradorRede(input_feature, camadas)

    def regressao(self, input_features, learning_rates, best_global_model = None):
        start = datetime.now()
        current_valid_loss = 0
        best_valid_loss = np.Inf
        train_losses, valid_losses = {}, {}

        for initializations in range(0, self.retries):
            for learning_rate in learning_rates:
                for batch_size in self.batch_sizes:

                    print(f'----------\nAtempt: {initializations}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\n')

                    model = self.model_generator(3, 500, 5)

                    if (best_global_model is not None):
                        model.load_state_dict(torch.load(best_global_model))
                    
                    criterion = nn.CrossEntropyLoss()
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

        return train_losses, valid_losses

################################################################################################

class Classificador():
    def __init__(self, dataset, epochs = 2000, batch_size = 25, early_stopping_epochs = 60):
        self.dataset               = dataset
        self.epochs                = epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.batch_size            = batch_size
    
    def model_generator(self, input_feature, neuronios, layers, p = .5, output_size = 2):
        camadas = []
        for i in range(0, layers):
            drp = random.uniform(0, p)
            c = Camada( int(neuronios/2**i), nn.ReLU(), nn.Dropout(drp) )
            camadas.append(c)

        camadas.append(  Camada( output_size, nn.Softmax(dim=-1) ) )
        return GeradorRede(input_feature, camadas)

    def classificacao(self):
        model = self.model_generator(3, 500, 5)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        t = Treinamento(self.dataset, self.epochs, self.batch_size, self.early_stopping_epochs)
        model, train_loss, valid_loss = t.train_linear(model, optimizer, criterion)

        return model, train_loss, valid_loss, t.train, t.valid


################################################################################################

def run_models(dataset, 
               run_epochs = 2000, run_early_stopping_epochs = 60,
               input_features = 3, learning_rates = [],
               run_retries = 5, run_batch_sizes = [],
               best_global_model = None):

  runner = Regressor(dataset,
                     epochs = run_epochs,
                     batch_sizes = run_batch_sizes,
                     early_stopping_epochs = run_early_stopping_epochs,
                     retries = run_retries )

  return runner.regressao(input_features, learning_rates, best_global_model) 

df = pd.read_csv('df_points.txt', sep='\t', index_col=[0])

input_features = 3
output_size = 2
p = 0.5

epochs = 2000
batch_size = 25
early_stopping_epochs = 50 # quantas épocas sem melhoria serão toleradas antes de parar o treinamento

train_losses, valid_losses = run_models(df,
                                        run_epochs = 5,
                                        run_early_stopping_epochs = 2,
                                        input_features = 3,
                                        learning_rates = [1e-4, 1e-5],
                                        run_retries = 1,
                                        run_batch_sizes = [10, 12])

#  r = Classificador(df, epochs, batch_size, early_stopping_epochs)
#  model, train_loss, valid_loss, train, valid = r.classificacao()
#
#  X_train, X_test, y_train, y_test = train
#  X_valid, X_test, y_valid, y_test = valid
#
#  p = Plots()
#  accuracy_final = p.get_accuracy(model, torch.from_numpy(X_test), torch.from_numpy(y_test.to_numpy()))
#  print(accuracy_final)
