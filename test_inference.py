from dotenv import load_dotenv

load_dotenv()

import os
import librosa
import soundata
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def load_wav_16k_mono(path):
    # path might be a 0-d NumPy array of type '<U..>' or bytes
    if isinstance(path, np.ndarray):
        path = path.item()
    if isinstance(path, (bytes, bytearray)):
        path = path.decode('utf-8')
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y.astype("float32")

def preprocess(file_path, label):
    wav = tf.numpy_function(load_wav_16k_mono, [file_path], tf.float32)
    wav.set_shape([None])            # let TF know it's 1-D
    wav = wav[:60000]
    padding = 60000 - tf.shape(wav)[0]
    wav = tf.pad(wav, [[0, padding]])

    stft = tf.signal.stft(wav, frame_length=320, frame_step=32)
    mag = tf.abs(stft)

    num_mel_bins = 64
    num_spectrogram_bins = mag.shape[-1]
    lower_hz, upper_hz = 80.0, 7600.0
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, lower_hz, upper_hz
    )
    mel = tf.matmul(mag, linear_to_mel)

    log_mel = tf.math.log(mel + 1e-6)[..., tf.newaxis]

    return log_mel, label


# ----- LOAD MODEL -----

print("Loading model and dataset...")
model = load_model("models\\siren_detector.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.Accuracy()]
)

# ----- GET DATASET -----

dataset = soundata.initialize('urbansound8k', data_home="urbansound8k")
# dataset.download()
# dataset.validate()

correctCounter = 0
incorrectCounter = 0

# ----- DO INFERENCE -----

for i in range(1000):

    # resolve the path of the file about to be predicted
    clip = dataset.choice_clip()
    fold = clip.fold
    id = clip.clip_id
    path = os.path.join("urbansound8k", "audio", f"fold{fold}", f"{id}.wav")
    class_label = clip.class_label
    print(f"\nPredicting file {path}...")

    spec, _ = preprocess(path, label=0)  # label here does not matter as it is ignored at inference
    spec = tf.expand_dims(spec, axis=0)

    yhat = model.predict(spec)
    probability = float(yhat[0, 0])
    prediction = "siren" if probability > 0.5 else "not siren"
    correct = "True" if prediction == class_label == "siren" else "True" if prediction != class_label and prediction != "siren" else "False"

    if correct == "True":
        correctCounter += 1

    else:
        incorrectCounter += 1

    print(f"\nFor file {path}:\n  Actual label: {class_label}\n  Predicted label: {prediction}\n  Certainty: {probability}\n  Correct: {correct}")
    print(f"Correct: {correctCounter} / Incorrect: {incorrectCounter},  average correct: {correctCounter/(i+1)}")
    # input("\nPress enter to continue...")

