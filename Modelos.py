import torch.nn as nn
import torch.nn.functional as F


class MinhaRede(nn.Module):
    def __init__(self, input_features, p=0.5):
        super(MinhaRede, self).__init__()

        self.camada_entrada = nn.Linear(input_features, 100)
        self.camada_oculta_1 = nn.Linear(100, 50)
        self.camada_oculta_2 = nn.Linear(50, 10)
        self.camada_saida = nn.Linear(10, 2)
        
        self.dropout_1 = nn.Dropout(p) # <= criação da camada de dropout, com cada neurônio tendo probabilidade p de ser desativado
        self.dropout_2 = nn.Dropout(p) # <= criação da camada de dropout, com cada neurônio tendo probabilidade p de ser desativado

    def forward(self, p):
        s = F.relu(self.camada_entrada(p))
        s = self.dropout_1(s) # <= aplicamos a camada de dropout, que só faz efeito quando o modelo está em modo de treinamento
        s = F.relu(self.camada_oculta_1(s))
        s = self.dropout_2(s) # <= aplicamos a camada de dropout, que só faz efeito quando o modelo está em modo de treinamento
        s = F.relu(self.camada_oculta_2(s))
        s = self.camada_saida(s)

        return s

class MinhaNovaRede(nn.Module):
    def __init__(self, input_features, p=0.5):
        super(MinhaNovaRede, self).__init__()

        self.camada_entrada = nn.Linear(input_features, 100)
        self.camada_oculta_1 = nn.Linear(100, 50)
        self.camada_oculta_2 = nn.Linear(50, 10)
        self.camada_saida = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        self.dropout_1 = nn.Dropout(p) # <= criação da camada de dropout, com cada neurônio tendo probabilidade p de ser desativado
        self.dropout_2 = nn.Dropout(p) # <= criação da camada de dropout, com cada neurônio tendo probabilidade p de ser desativado

    def forward(self, p):
        s = self.relu(self.camada_entrada(p))
        s = self.dropout_1(s) # <= aplicamos a camada de dropout, que só faz efeito quando o modelo está em modo de treinamento
        s = self.relu(self.camada_oculta_1(s))
        s = self.dropout_2(s) # <= aplicamos a camada de dropout, que só faz efeito quando o modelo está em modo de treinamento
        s = F.relu(self.camada_oculta_2(s))
        s = self.softmax(self.camada_saida(s))

        return s
