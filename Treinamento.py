import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime


class Treinamento():
    def get_batches(self, data, batch_size=1):
        batches = []
        
        data_size = len(data)
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(data_size, start_idx + batch_size)
            batches.append(data[start_idx:end_idx])
        
        return batches

    def train(self, model, n_epochs, batch_size, early_stopping_epochs, optimizer, criterion, X_train, y_train, X_valid, y_valid):
        init = datetime.now()
        
        best_epoch = None
        best_valid_loss = np.Inf
        best_train_loss = None
        epochs_without_improv = 0
        
        train_loss = []
        valid_loss = []
    
        for epoch in tqdm(range(n_epochs)):
            ###################
            # early stopping? #
            ###################
            if epochs_without_improv >= early_stopping_epochs:
                break
            
            ###################
            # train the model #
            ###################
            model.train()
            acc_train_loss = 0.0
            for index, (original_data, original_target) in enumerate(zip(self.get_batches(X_train, batch_size),
                                                                         self.get_batches(y_train, batch_size))):
                
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
    
            train_loss.append(acc_train_loss)
    
            ###################
            # valid the model #
            ###################
            model.eval()
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
    
            valid_loss.append(acc_valid_loss)
            
            #####################
            # Update best model #
            #####################
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
