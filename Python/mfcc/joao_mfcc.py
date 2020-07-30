from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

def main():
    (rate,sig) = wav.read("..\\segmentacao\\resultado.wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)

    print(len(fbank_feat)) #[1:3,:]

    # plt.figure(figsize=(15,5))
    plt.imshow(fbank_feat[0:300:,0:13], aspect='auto', origin='lower')
    plt.show()
    


main()