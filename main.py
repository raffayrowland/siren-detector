from dotenv import load_dotenv

load_dotenv()

import soundata
import librosa
import tensorflow as tf
# from matplotlib import pyplot as plt
import os

def load_wav_16k_mono(path):
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y

dataset = soundata.initialize('urbansound8k', data_home="C:\\Users\\raffa\\PycharmProjects\\siren-detector")

dataDict = dataset.load_clips()

print(dataDict)