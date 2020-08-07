import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt

# from scipy import misc
# import numpy as np
import pickle
########################################
import os


def load_mfcc_GPU():
    lista = os.listdir("..\\mfcc\\resultados")
    retorno = []
    retorno_gabarito = []

    ######################################pega os mfcc
    for i in lista:
        with open("..\\mfcc\\resultados\\" + i, "rb") as fp:
        	retorno.append(pickle.load(fp))
    retorno = torch.cuda.FloatTensor(retorno)
    # print(len(retorno[0]))

    ################################## gabarito        
    arquivo = open("REFERENCE.csv", 'r')
    
    linha = arquivo.readlines()

    for i in linha:
        i = i.replace('\n', '')
        i = i.split(',')[1]
        if int(i) == 1:
            retorno_gabarito.append(True)
        else:
            retorno_gabarito.append(False)

    retorno_gabarito = torch.cuda.BoolTensor(retorno_gabarito)
    # print(len(retorno[0][0]))
    
    
    # print(retorno)
    return retorno, retorno_gabarito



class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CV0 = nn.Conv2d(in_channels=1800, out_channels=1800, kernel_size=(2,20))
        self.MXP0 = nn.MaxPool2d(kernel_size=(1, 20), stride=5)

        self.CV1 = nn.Conv2d(in_channels=360, out_channels=360, kernel_size=(2,10))
        self.MXP1 = nn.MaxPool2d(kernel_size=(1, 4), stride=2)

        self.FLA = nn.Flatten()

        self.output0 = nn.Linear(in_features=1024, out_features=512)
        self.output1 = nn.Linear(in_features=512, out_features=2)


    def forward(self, x):
        x = F.relu(self.CV0(x))
        x = F.relu(self.MXP0(x))
        x = F.relu(self.CV1(x))
        x = F.relu(self.MXP1(x))
        x = F.sigmoid(self.output0(x))
        x = F.sigmoid(self.output1(x))
        # x = self.output1(x)
        return x