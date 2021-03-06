import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
########################################
import os


def load_mfcc_GPU():
    lista = os.listdir("..\\..\\..\\Treino")
    retorno = []
    retorno_gabarito = []

    ######################################pega os mfcc
    for i in lista[:-1]:
        with open("..\\..\\..\\Treino\\" + i, "rb") as fp:
            retorno.append(pickle.load(fp))

    retorno = torch.cuda.FloatTensor(retorno)
    retorno.requires_grad_()

    ################################## gabarito        
    arquivo = open("..\\..\\..\\Treino\\REFERENCE_treino.csv", 'r')
    
    linha = arquivo.readlines()

    for i in linha:
        # print(i)
        # input()
        i = i.replace('\n', '')
        i = i.split(',')[1]

        if int(i) == -1:
            retorno_gabarito.append(0)
        else:
            retorno_gabarito.append(1)
        
        # break

    retorno_gabarito = torch.cuda.LongTensor(retorno_gabarito)
    torch.save(retorno, "data.pt")
    torch.save(retorno_gabarito, "target.pt")

class ANN(nn.Module):
    def __init__(self):
        # super().__init__()
        super(ANN, self).__init__()
        
        self.CV0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(20,1), padding_mode='reflect', padding=(10,0)) #o certo era 20,2
        self.MXP0 = nn.MaxPool2d(kernel_size=(20, 1), stride=(5,1), padding=(5,0))

        self.CV1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(10,1), padding_mode='reflect', padding=(5,0))
        self.MXP1 = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1), padding=(1,0))

        # self.FLA = nn.Flatten()

        self.output0 = nn.Linear(in_features=11520, out_features=512)

        self.output1 = nn.Linear(in_features=512, out_features=2)

        


    def forward(self, x):
        # print("inicio: " + str(x.shape))
        x = F.relu(self.CV0(x))
        # print("CV0: " + str(x.shape))
        x = F.relu(self.MXP0(x))
        # print("MXP0: " + str(x.shape))
        x = F.relu(self.CV1(x))
        # print("CV1: " + str(x.shape))
        x = F.relu(self.MXP1(x))
        # print("MXP1: " + str(x.shape))
        
        x = torch.flatten(x) #start_dim=0, end_dim=-1
        # print("tamanho flatten: " + str(len(x)))

        x = torch.sigmoid(self.output0(x))
        x = self.output1(x)

        return x

class treino():
    def __init__ (self):
        self.rede = None
        self.epoca = None
        self.erro_treino = None
        self.erro_validacao = None