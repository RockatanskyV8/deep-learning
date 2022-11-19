import torch
import torch.optim as optim

from Treinamento import Treinamento
from Plots import Plots

class Runner():
    def __init__(self, model, epochs, batch_size, early_stopping_epochs, retries = 5, batches = []):
        self.model                 = model
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.early_stopping_epochs = early_stopping_epochs # quantas épocas sem melhoria serão toleradas antes de parar o treinamento
        self.retries               = retries
        self.batches               = batches


    def classificacao(self, optimizer, criterion, X_train, y_train, X_valid, y_valid):
        t = Treinamento()
        model, train_loss, valid_loss = t.train(self.model,
                                                self.epochs,
                                                self.batch_size,
                                                self.early_stopping_epochs,
                                                optimizer,
                                                criterion,
                                                X_train,
                                                y_train,
                                                X_valid,
                                                y_valid)

        p = Plots()
        p.plot_losses(train_loss, valid_loss)

        return model


    def regressao(self, input_features, learning_rates, criterion, X_train, y_train, X_valid, y_valid):
        current_valid_loss = 0
        t = Treinamento()
        for initializations in range(0, self.retries):
            for lr in learning_rates:
                for batch_size in self.batches:

                    print(f'----------\nLearning rate: {lr}\nBatch size: {batch_size}\n')

                    model = self.model(input_features)
                    optimizer = optim.SGD(model.parameters(), lr=lr)

                    model, train_loss, valid_loss = t.train(model,
                                                            self.epochs,
                                                            batch_size,
                                                            self.early_stopping_epochs,
                                                            optimizer,
                                                            criterion,
                                                            X_train,
                                                            y_train,
                                                            X_valid,
                                                            y_valid)

                    # store best valid loss
                    current_valid_loss = min(valid_loss)
                    if current_valid_loss < best_valid_loss:
                        torch.save(model.state_dict(), 'best_global_model')
                        best_valid_loss = current_valid_loss
                        print('New best global model found!')

                    print(f'\nValidation loss: {current_valid_loss}\n')

            p = Plots()
            p.get_loss(model,
                       torch.from_numpy(X_test.to_numpy()),
                       torch.from_numpy(y_test),
                       criterion)

            return current_valid_loss

