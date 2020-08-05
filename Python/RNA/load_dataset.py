import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy import misc
import numpy as np
import pickle
########################################
import os


def load_mfcc_GPU ():
    lista = os.listdir("..\mfcc\\resultados")
    retorno = []
    retorno_gabarito = []

    ######################################pega os mfcc
    for i in lista:
        with open("..\mfcc\\resultados\\" + i, "rb") as fp:
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
    print(len(retorno_gabarito))
    
    
    # print(retorno)
    return retorno, retorno_gabarito


load_mfcc_GPU()