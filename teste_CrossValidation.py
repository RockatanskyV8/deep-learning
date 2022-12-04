import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import SubsetRandomSampler, DataLoader

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold

from Reader import *


class Treinamento():
    def get_batches(self, data, batch_size=1):
        batches = []
        
        data_size = len(data)
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(data_size, start_idx + batch_size)
            batches.append(data[start_idx:end_idx])
        
        return batches

    def reset_weights(m):
        # Try resetting model weights to avoid weight leakage.
        for layer in m.children():
         if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()

    def cross_train(self, X_train, y_train, model, optimizer, criterion, batch_size = 1):
        acc_train_loss = 0.0

        for index, (original_data, original_target) in enumerate(zip(self.get_batches(X_train, batch_size),
                                                                     self.get_batches(y_train, batch_size))):
                
                print(original_target)
                print((original_target == 1))
                print()
                # Format data to tensor
                target = (original_target == 1).nonzero(as_tuple=True)[1]
                data = original_data.float() # Esse '.float()' é necessário para arrumar o tipo do dado

                # target = target.cuda()
                # data = data.cuda()

                optimizer.zero_grad()

                # model.forward(data)
                predicted = model(data)

                loss = criterion(predicted, target)

                # Backprop
                loss.backward()
                optimizer.step()

                acc_train_loss += loss.item()

        return acc_train_loss

    def cross_valid(self, X_valid, y_valid, model, criterion, batch_size = 1):
        acc_valid_loss = 0.0

        for index, (original_data, original_target) in enumerate(zip(self.get_batches(X_valid, batch_size), 
                                                                     self.get_batches(y_valid, batch_size))):
            # Format data to tensor
            target = (original_target == 1).nonzero(as_tuple=True)[1]
            data = original_data.float() # Esse '.float()' é necessário para arrumar o tipo do dado
    
            # target = target.cuda()
            # data = data.cuda()
    
            # model.forward(data)
            predicted = model(data)
    
            loss = criterion(predicted, target)
            acc_valid_loss += loss.item()
    
        return acc_valid_loss

    def train(self, model, n_epochs, batch_size, early_stopping_epochs, optimizer, criterion, dataset):
        init = datetime.now()
        
        best_epoch = None
        best_valid_loss = np.Inf
        best_train_loss = None
        epochs_without_improv = 0

        train_loss = []
        valid_loss = []

        read = Reader(dataset)
        train, valid = read.read()

        X_train, X_test, y_train, y_test = train
        X_valid, X_test, y_valid, y_test = valid

        df = zip(self.get_batches(X_train, batch_size),
                 self.get_batches(y_train, batch_size))

        kf = KFold(n_splits=4, random_state=1, shuffle=True)
        split = kf.split(train)
        #  print(type(split))
        for idx, (train_idx, valid_idx) in enumerate(split):
            print('Index {}'.format(idx + 1))

            y_cros_train, y_cros_valid = y_train.iloc[train_idx], y_test.iloc[valid_idx]
            X_cros_train, X_cros_valid = X_train[train_idx,:],    X_test[valid_idx,:]

            for epoch in tqdm(range(n_epochs)):

                if epochs_without_improv >= early_stopping_epochs:
                    break

                model.train()
                acc_train_loss = self.cross_train(torch.from_numpy(X_cros_train), torch.from_numpy(y_cros_train.to_numpy()), 
                                                  model, optimizer, criterion, batch_size)
                train_loss.append(acc_train_loss)

                model.eval()
                acc_valid_loss = self.cross_valid(torch.from_numpy(X_cros_valid), torch.from_numpy(y_cros_valid.to_numpy()), 
                                                  model, criterion, batch_size)
                valid_loss.append(acc_valid_loss)

                if acc_valid_loss < best_valid_loss:
                    torch.save(model.state_dict(), 'best_model') # save best model
                    best_epoch = epoch
                    best_valid_loss = acc_valid_loss
                    best_train_loss = acc_train_loss
                    epochs_without_improv = 0
                else:
                    epochs_without_improv += 1

        # Load best model
        model.load_state_dict(torch.load('best_model'))
        model.eval()

        # Print logs
        if epochs_without_improv >= early_stopping_epochs:
            print('Training interrupted by early stopping!')
        else:
            print('Training finished by epochs!')
        print(f'Total epochs run: {epoch + 1}')
        print(f'Best model found at epoch {best_epoch + 1} with valid loss {best_valid_loss} and training loss {best_train_loss}')

        end = datetime.now()
        print(f'Total training time: {end - init}')

        return model, train_loss, valid_loss

