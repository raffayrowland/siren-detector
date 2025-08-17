from dotenv import load_dotenv

load_dotenv()

import librosa
import tensorflow as tf

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

# ----- PROCESS DATA ------

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
    0: total / (2 * negativeCount) * 3,  # Penalise false positives
    1: total / (2 * positiveCount),
}

print(f"Negative weight: {classWeight[0]},  Positive weight: {classWeight[1]}")

data = positives.concatenate(negatives) # concatenate them to one dataset

data = data.map(preprocess)
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# define the training and testing partitions
train = data.take(int(len(data) * 0.7))
test = data.skip(int(len(data) * 0.7)).take(len(data) - int(len(data) * 0.7))
del data
train = train.cache()
test = test.cache()

print(f"Training batches: {len(train)}, Testing batches: {len(test)}")

# ----- TRAIN -----
from tensorflow_model_optimization.python.core.keras.compat import keras

# define the model's layers
model = keras.Sequential([
  keras.layers.Input(shape=(1866, 257, 1)),
  keras.layers.Conv2D(16, (3, 3), activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Recall(), keras.metrics.Precision()])

# Train the model
hist = model.fit(train, epochs=1, validation_data=test, class_weight=classWeight)

loss, recall, precision = model.evaluate(test)
print(f"Baseline stats:  Loss: {loss}, Precision: {precision}, Recall: {recall}")

# ----- SAVE MODEL -----
from tensorflow.keras import backend as K

model.save_weights("weights\\pretrained_weights.h5")

try:
    model.save("models/siren_detector.keras")
    print("Saved model in keras format")

except ValueError as e:
    print(e)
    model.save("models\\siren_detector.h5")
    print("Saved model")

# Clear unused stuff from memory
del model, train, test, hist
K.clear_session()

# ----- PRUNE MODEL -----

import tensorflow_model_optimization as tfmot

EPOCHS = 1

# Redefine the base model for pruning
baseModel = keras.Sequential([
  keras.layers.Input(shape=(1866, 257, 1)),
  keras.layers.Conv2D(16, (3, 3), activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid'),
])

# Build a more memory-friendly dataset for pruning
data = positives.concatenate(negatives)
data = data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
data = data.shuffle(buffer_size=325)
data = data.batch(4)
data = data.prefetch(1)

# Define the new training and test datasets
train = data.take(int(len(data) * 0.7))
test = data.skip(int(len(data) * 0.7)).take(len(data) - int(len(data) * 0.7))

# Load pretrained model weights and set parameters for pruning
baseModel.load_weights("weights\\pretrained_weights.h5")
steps_per_epoch = int(tf.data.experimental.cardinality(train).numpy())
pruning_params = {
  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
     initial_sparsity=0.0,
     final_sparsity=0.5,
     begin_step=0,
     end_step=steps_per_epoch * EPOCHS
  )
}
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

modelForPruning = tfmot.sparsity.keras.prune_low_magnitude(baseModel, **pruning_params)
modelForPruning.summary()
modelForPruning.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Recall(), keras.metrics.Precision()])

# Train the model
modelForPruning.fit(train, epochs=EPOCHS, validation_data=test, callbacks=callbacks)
del callbacks

# Strip the pruning
modelForExport = tfmot.sparsity.keras.strip_pruning(modelForPruning)
del modelForPruning
modelForExport.summary()
modelForExport.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Recall(), keras.metrics.Precision()])

# Evaluate the model
loss, recall, precision = modelForExport.evaluate(test)
print(f"Pruned stats:  Loss: {loss}, Precision: {precision}, Recall: {recall}")

# Save the model
modelForExport.save("models\\pruned_model.h5", include_optimizer=True)