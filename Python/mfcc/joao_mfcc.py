from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(file):
    (rate,sig) = wav.read("..\\segmentacao\\resultados\\" + file)
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)

    # print(len(fbank_feat)) #[1:3,:]

    # plt.figure(figsize=(15,5))
    plt.imshow(fbank_feat[0:301:,0:13], aspect='auto', origin='lower')
    plt.savefig("..\\mfcc\\resultados\\"+file[:-3]+"png")
    # plt.show()
    


main(sys.argv[1])