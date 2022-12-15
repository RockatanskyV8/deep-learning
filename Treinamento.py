import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import KFold

from Reader import *
from CustomDataset import *

class Treinamento():

    def __init__(self, dataset, 
                       n_epochs=10,
                       batch_size=1, 
                       early_stopping_epochs=10):

        read = Reader(dataset)
        self.train, self.valid = read.read()

        self.X_train, self.X_test, self.y_train, self.y_test = self.train
        self.X_valid, self.X_test, self.y_valid, self.y_test = self.valid

        transformations = transforms.Compose([ transforms.ToTensor() ])

        self.train_set = CustomDataset(self.X_train, self.y_train)
        self.valid_set = CustomDataset(self.X_valid, self.y_valid)
        self.test_set  = CustomDataset(self.X_test,  self.y_test)

        self.n_epochs              = n_epochs
        self.batch_size            = batch_size           
        self.early_stopping_epochs = early_stopping_epochs

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size)
        self.test_loader  = torch.utils.data.DataLoader(self.test_set,  batch_size=self.batch_size)

    # UTILS
    def get_batches(self, data, batch_size=1):
        batches = []
        
        data_size = len(data)
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(data_size, start_idx + batch_size)
            batches.append(data[start_idx:end_idx])
        
        return batches

    def load_best_model(self, model, epoch, best_epoch, best_valid_loss, best_train_loss, epochs_without_improv):
        # Load best model
        model.load_state_dict(torch.load('best_model'))
        model.eval()
        
        # Print logs
        if epochs_without_improv >= self.early_stopping_epochs:
            print('Training interrupted by early stopping!')
        else:
            print('Training finished by epochs!')
        print(f'Total epochs run: {epoch + 1}')
        print(f'Best model found at epoch {best_epoch + 1} with valid loss {best_valid_loss} and training loss {best_train_loss}')


    ###############################################################################################################################
    #################################### LOSSES ###################################################################################
    ###############################################################################################################################

    def train_loss(self, X_train, y_train, zip_batch, optimizer, criterion, model):
        model.train()
        acc_train_loss = 0.0
        total_batches = 0
        for index, (original_data, original_target) in enumerate(zip_batch):
            total_batches += 1
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

        return total_batches, acc_train_loss

    def valid_loss(self, X_valid, y_valid, zip_batch, criterion, model):
        model.eval()
        acc_valid_loss = 0.0
        total_batches = 0
        for index, (original_data, original_target) in enumerate(zip_batch):
            total_batches += 1
            # Format data to tensor
            target = (original_target == 1).nonzero(as_tuple=True)[1]
            data = original_data.float() # Esse '.float()' é necessário para arrumar o tipo do dado
    
            # target = target.cuda()
            # data = data.cuda()
    
            # model.forward(data)
            predicted = model(data)
    
            loss = criterion(predicted, target)
            acc_valid_loss += loss.item()
    
        return total_batches, acc_valid_loss

    ###############################################################################################################################
    #################################### TREINAMENTOS #############################################################################
    ###############################################################################################################################

    def train_linear(self, model, optimizer, criterion):
        init = datetime.now()
        
        best_epoch = None
        best_valid_loss = np.Inf
        best_train_loss = None
        epochs_without_improv = 0
        
        train_loss = []
        valid_loss = []
    
        for epoch in tqdm(range(self.n_epochs)):
            ###################
            # early stopping? #
            ###################
            if epochs_without_improv >= self.early_stopping_epochs:
                break
            
            ###################
            # train the model #
            ###################
            X_train1, y_train1 = torch.from_numpy(self.X_train), torch.from_numpy(self.y_train.to_numpy())
            zip_batch_train    = zip(self.get_batches(X_train1, self.batch_size), self.get_batches(y_train1, self.batch_size))

            _, acc_train_loss = self.train_loss(X_train1, y_train1,
                                                zip_batch_train,
                                                optimizer,
                                                criterion,
                                                model)
            train_loss.append(acc_train_loss)
    
            ###################
            # valid the model #
            ###################
            X_valid1, y_valid1 = torch.from_numpy(self.X_valid), torch.from_numpy(self.y_valid.to_numpy())
            zip_batch_valid    = zip(self.get_batches(X_valid1, self.batch_size), self.get_batches(y_valid1, self.batch_size))

            _, acc_valid_loss = self.valid_loss(X_valid1, y_valid1,
                                                zip_batch_valid,
                                                criterion,
                                                model)
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
        
        self.load_best_model(model, epoch, best_epoch, best_valid_loss, best_train_loss, epochs_without_improv)
        end = datetime.now()
        print(f'Total training time: {end - init}')
        
        return model, train_loss, valid_loss


    def train_cross_validation(self, model, optimizer, criterion):
        init = datetime.now()
        
        best_epoch = None
        best_valid_loss = np.Inf
        best_train_loss = None
        epochs_without_improv = 0

        train_loss = []
        valid_loss = []

        kf = KFold(n_splits=4, random_state=1, shuffle=True)
        
        for idx, (train_idx, valid_idx) in enumerate( kf.split(self.train) ):
            print('Index {}'.format(idx + 1))

            y_cros_train, y_cros_valid = self.y_train.iloc[train_idx], self.y_valid.iloc[valid_idx]
            X_cros_train, X_cros_valid = self.X_train[train_idx,:],    self.X_valid[valid_idx,:]

            for epoch in tqdm(range(self.n_epochs)):

                ###################
                # train the model #
                ###################
                X_train1, y_train1 = torch.from_numpy(X_cros_train), torch.from_numpy(y_cros_train.to_numpy())
                zip_batch_train    = zip(self.get_batches(X_train1, self.batch_size), self.get_batches(y_train1, self.batch_size))

                _, acc_train_loss = self.train_loss(X_train1, y_train1,
                                                    zip_batch_train,
                                                    optimizer,
                                                    criterion,
                                                    model)
                train_loss.append(acc_train_loss)
                
                ###################
                # valid the model #
                ###################
                X_valid1, y_valid1 = torch.from_numpy(X_cros_valid), torch.from_numpy(y_cros_valid.to_numpy())
                zip_batch_valid    = zip(self.get_batches(X_valid1, self.batch_size), self.get_batches(y_valid1, self.batch_size))

                _, acc_valid_loss = self.valid_loss(X_valid1, y_valid1,
                                                    zip_batch_valid,
                                                    criterion,
                                                    model)
                valid_loss.append(acc_valid_loss)

                if acc_valid_loss < best_valid_loss:
                    torch.save(model.state_dict(), 'best_model') # save best model
                    best_epoch = epoch
                    best_valid_loss = acc_valid_loss
                    best_train_loss = acc_train_loss
                    epochs_without_improv = 0
                else:
                    epochs_without_improv += 1

        self.load_best_model(model, epoch, best_epoch, best_valid_loss, best_train_loss, epochs_without_improv)
        end = datetime.now()
        print(f'Total training time: {end - init}')
        
        return model, train_loss, valid_loss

    def train_cross_validation_dataloader(self, model, optimizer, criterion):
        init = datetime.now()
        
        best_epoch = None
        best_valid_loss = np.Inf
        best_train_loss = None
        epochs_without_improv = 0
        
        train_loss = []
        valid_loss = []

        for epoch in tqdm(range(self.n_epochs)):
            ###################
            # early stopping? #
            ###################
            if epochs_without_improv >= self.early_stopping_epochs:
                break
            
            ###################
            # train the model #
            ###################
            X_train1, y_train1 = torch.from_numpy(self.X_train), torch.from_numpy(self.y_train.to_numpy())

            total_batches, acc_train_loss = self.train_loss(X_train1, y_train1,
                                                            self.train_loader,
                                                            optimizer,
                                                            criterion,
                                                            model)
            train_loss.append(acc_train_loss/total_batches)
    
            ###################
            # valid the model #
            ###################
            X_valid1, y_valid1 = torch.from_numpy(self.X_valid), torch.from_numpy(self.y_valid.to_numpy())

            total_batches, acc_valid_loss = self.valid_loss(X_valid1, y_valid1,
                                                            self.valid_loader,
                                                            criterion,
                                                            model)
            valid_loss.append(acc_valid_loss/total_batches)

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
        
        self.load_best_model(model, epoch, best_epoch, best_valid_loss, best_train_loss, epochs_without_improv)
        end = datetime.now()
        print(f'Total training time: {end - init}')
        
        return model, train_loss, valid_loss
