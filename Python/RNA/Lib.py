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
    # retorno.requires_grad_()

    ################################## gabarito        
    arquivo = open("REFERENCE.csv", 'r')
    
    linha = arquivo.readlines()

    for i in linha:
        i = i.replace('\n', '')
        i = i.split(',')[1]

        if int(i) == -1:
            retorno_gabarito.append(0)
        else:
            retorno_gabarito.append(1)
        
        # break

    retorno_gabarito = torch.cuda.FloatTensor(retorno_gabarito)
    # retorno_gabarito.requires_grad_()
    
    
    return retorno, retorno_gabarito



class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.CV0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(1,1), bias=True)
        self.MXP0 = nn.MaxPool2d(kernel_size=(1, 1), stride=(5,1))

        self.CV1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), bias=True)
        self.MXP1 = nn.MaxPool2d(kernel_size=(1,1), stride=(2,1))

        # self.FLA = nn.Flatten()

        self.output0 = nn.Linear(in_features=11520, out_features=512)
        self.output1 = nn.Linear(in_features=512, out_features=1)
        


    def forward(self, x):
        x = F.relu(self.CV0(x))
        print(x.shape)
        x = F.relu(self.MXP0(x))
        print(x.shape)
        x = F.relu(self.CV1(x))
        print(x.shape)
        x = F.relu(self.MXP1(x))
        print(x.shape)
        
        x = torch.flatten(x) #start_dim=0, end_dim=-1
        print(len(x))

        x = torch.sigmoid(self.output0(x))
        x = self.output1(x)

        return x