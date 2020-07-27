from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def main():
    (rate,sig) = wav.read("..\\..\\Banco_A\\a0001.wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)

    print(len(fbank_feat[0])) #[1:3,:]

main()