from dotenv import load_dotenv

load_dotenv()

import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from matplotlib import pyplot as plt
import os

def load_wav_16k_mono(path):
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y.astype("float32")

def preprocess(file_path, label):
    wav = tf.numpy_function(load_wav_16k_mono, [file_path], tf.float32)
    wav.set_shape([None])            # let TF know it's 1-D
    wav = wav[:60000]
    padding = 60000 - tf.shape(wav)[0]
    wav = tf.pad(wav, [[0, padding]])
    spec = tf.signal.stft(wav, 320, 32)
    spec = tf.abs(spec)[..., tf.newaxis]
    return spec, label

# directories for the positive and negative training files
POSITIVE = "data\\positive"
NEGATIVE = "data\\negative"

# gets a list of wav files in those directories
pos = tf.data.Dataset.list_files(POSITIVE + '\*.wav')
neg = tf.data.Dataset.list_files(NEGATIVE + '\*.wav')

# turn those lists into datasets, and add the 1/0 labels for siren/no siren
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.zeros(len(pos)))))

# There are considerably more non-siren examples than siren examples, so this weights each class to make it fair
positiveCount = tf.data.experimental.cardinality(positives).numpy()
negativeCount= tf.data.experimental.cardinality(negatives).numpy()
total =  positiveCount + negativeCount
classWeight = {
    0: total / (2 * negativeCount),
    1: total / (2 * positiveCount),
}

data = positives.concatenate(negatives) # concatenate them to one dataset

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# define the training and testing partitions
train = data.take(6112)
test = data.skip(6112).take(2620)

# define the model's layers
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1866, 257, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile("Adam", loss="BinaryCrossentropy", metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

# Train the model
hist = model.fit(train, epochs=4, validation_data=test, class_weight=classWeight)