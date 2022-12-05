import torch
import torch.optim as optim

import numpy as np
from Treinamento import Treinamento
from Modelos import *
from Plots import *

from datetime import datetime

class Classificador():
    def __init__(self, dataset, model, epochs = 2000, batch_size = 25, early_stopping_epochs = 60):
        self.dataset               = dataset
        self.model                 = model
        self.epochs                = epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.batch_size            = batch_size

    def classificacao(self, optimizer, criterion):
        t = Treinamento(self.dataset, self.epochs, self.batch_size, self.early_stopping_epochs)
        model, train_loss, valid_loss = t.train_linear(self.model, optimizer, criterion)

        return model, train_loss, valid_loss, t.train, t.valid

class Validacao_Cruzada():
    def __init__(self, dataset, model, epochs = 2000, batch_size = 25, early_stopping_epochs = 60):
        self.dataset               = dataset
        self.model                 = model
        self.epochs                = epochs
        self.early_stopping_epochs = early_stopping_epochs
        self.batch_size            = batch_size
    
    def classificacao(self, optimizer, criterion):
        t = Treinamento(self.dataset, self.epochs, self.batch_size, self.early_stopping_epochs)
        model, train_loss, valid_loss = t.train_cross_validation(self.model, optimizer, criterion)

        return model, train_loss, valid_loss, t.train, t.valid

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

