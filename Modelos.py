import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

class Camada():
    def __init__(self, size, activation = None, dropout = None):
        self.size = size
        self.activation = activation
        self.dropout = dropout

class GeradorRede(nn.Module):
    def __init__(self, input_size, layers_data: list):
        super(GeradorRede, self).__init__()
        
        self.layers = nn.ModuleList()
        self.input_size = input_size

        for layer in layers_data:
            self.layers.append(nn.Linear(input_size, layer.size))
            input_size = layer.size

            if layer.activation is not None:
                assert isinstance(layer.activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(layer.activation)

            if layer.dropout is not None:
                self.layers.append( layer.dropout )

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

