import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

class GeradorRede(nn.Module):
    def __init__(self, input_size, layers_data: list, dropouts: list, p=0.5):
        super(GeradorRede, self).__init__()
        
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.p = p

        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

        for dropout in dropouts:
            self.layers.append(dropout)
            

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

