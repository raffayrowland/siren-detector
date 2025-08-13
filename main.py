from dotenv import load_dotenv

load_dotenv()

import soundata
import librosa
import tensorflow as tf
from matplotlib import pyplot as plt
import os

def load_wav_16k_mono(path):
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:60000]
    zero_padding = tf.zeros([60000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# directories for the positive and negative training files
POSITIVE = "data\\train\\positive"
NEGATIVE = "data\\train\\negative"

# gets a list of wav files in those directories
pos = tf.data.Dataset.list_files(POSITIVE + '\*.wav')
neg = tf.data.Dataset.list_files(NEGATIVE + '\*.wav')

# turn those lists into datasets, and add the 1/0 labels for siren/no siren
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.zeros(len(pos)))))
data = positives.concatenate(negatives) # concatenate them to one dataset

# get a list of all the lengths of the wav files
lengths = []
for file in os.listdir("data\\train\\positive"):
    tensorWave = load_wav_16k_mono(os.path.join("data\\train\\positive", file))
    lengths.append(len(tensorWave))

filepath, label = negatives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
print(spectrogram)

plt.figure(figsize=(30, 20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()