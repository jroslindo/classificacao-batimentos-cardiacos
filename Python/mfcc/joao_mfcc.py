from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import pickle

def main(file):
    (rate,sig) = wav.read("..\\segmentacao\\resultados\\" + file)
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)

    # utilizar para salvar o vetor de mfcc
    with open("..\\mfcc\\resultados\\"+file[:-3]+"txt", "wb") as fp:
        pickle.dump(fbank_feat[0:301:,0:6], fp)

    

    
    # x = torch.cuda.FloatTensor(fbank_feat[0:301:,0:13])
    # torch.save(x, "..\\mfcc\\resultados\\"+file[:-3]+"pt")
    

    #utilizar caso queira ver os mfcc como imagens
    # plt.imshow(fbank_feat[0:301:,0:13], aspect='auto', origin='lower')
    # plt.savefig("..\\mfcc\\resultados\\"+file[:-3]+"png")
    # plt.show()
    


main(sys.argv[1])