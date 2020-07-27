from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def main():
    (rate,sig) = wav.read("..\\segmentacao\\resultado.wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)

    print(fbank_feat) #[1:3,:]

main()