from dotenv import load_dotenv

load_dotenv()

import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input

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
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

# There are considerably more non-siren examples than siren examples, so this weights each class to make it fair
positiveCount = tf.data.experimental.cardinality(positives).numpy()
negativeCount = tf.data.experimental.cardinality(negatives).numpy()
total = positiveCount + negativeCount
print(f"Positive samples: {positiveCount},  Negative samples: {negativeCount},  Total samples: {total}")
classWeight = {
    0: total / (2 * negativeCount) * 3,
    1: total / (2 * positiveCount),
}

print(f"Negative weight: {classWeight[0]},  Positive weight: {classWeight[1]}")

data = positives.concatenate(negatives) # concatenate them to one dataset

data = data.map(preprocess)
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# define the training and testing partitions (commented cache to save on ram)
train = data.take(int(len(data) * 0.7))
test = data.skip(int(len(data) * 0.7)).take(len(data) - int(len(data) * 0.7))
# train = train.cache()
# test = test.cache()

print(len(train), len(test))

# define the model's layers
model = Sequential()
model.add(Input(shape=(1866, 257, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile("Adam", loss="BinaryCrossentropy", metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

# Train the model
hist = model.fit(train, epochs=5, validation_data=test, class_weight=classWeight)

try:
    model.save('models\\my_model.keras')
    print("Saved model")

except ValueError:
    model.save("models\\siren_detector.h5")
    print("Saved model")


xtest, ytest = test.as_numpy_iterator().next()

yhat = model.predict(xtest)

yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
print(yhat)
print(ytest.astype(int))