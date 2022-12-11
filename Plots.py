import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px

import torch
import torch.nn.functional as F

class Plots():

    def plot_losses(self, train_loss, valid_loss):
        figure(figsize=(12, 8))
        
        # Plot and label the training and validation loss values
        epochs = len(train_loss)
        plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs+1), valid_loss, label='Validation Loss')
        
        # Add in a title and axes labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        # Display the plot
        plt.legend(loc='best')
        plt.show()

    def plot_losses(self, train_loss, valid_loss):
        figure(figsize=(12, 8))
        
        # Plot and label the training and validation loss values
        epochs = len(train_loss)
        plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs+1), valid_loss, label='Validation Loss')
   
        # Add in a title and axes labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
   
        # Display the plot
        plt.legend(loc='best')
        plt.show()

    def get_accuracy(self, model, X_test, y_test):
        model.eval()
    
        hits = 0
        for index, (original_data, original_target) in enumerate(zip(X_test, y_test)):
            # Format data to tensor
            target = original_target
            data = torch.tensor(()).new_ones((1, 3))
            data[0] = original_data
    
            # Softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
            # Probability for each output
            predicted = F.softmax(model(data), dim=1)
    
            # The output with the highest probability is the predicted class
            # Let's calculate the accuracy
            if torch.argmax(predicted[0]) == torch.argmax(target):
                hits += 1
                
        return hits/(index+1)
    
    def get_loss(self, model, X_test, y_test, criterion):
        model.eval()
        predicted = model(X_test.float())
        loss = criterion(predicted, y_test.float())
        return loss.item()
