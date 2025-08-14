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
    spec = tf.signal.stft(wav, 320, 32)
    spec = tf.abs(spec)[..., tf.newaxis]
    return spec, label

print("Loading model and dataset...")
model = load_model("models/siren_detector.keras")
dataset = soundata.initialize('urbansound8k', data_home="C:\\Users\\raffa\\PycharmProjects\\siren-detector")
# dataset.download()
# dataset.validate()

while True:
    clip = dataset.choice_clip()
    fold = clip.fold
    id = clip.clip_id
    path = os.path.join("audio", f"fold{fold}", f"{id}.wav")
    class_label = clip.class_label
    print(f"\nPredicting file {path}...")

    spec, _ = preprocess(path, label=0)  # label here does not matter as it is ignored at inference

    spec = tf.expand_dims(spec, axis=0)

    yhat = model.predict(spec)
    probability = float(yhat[0, 0])
    prediction = "siren" if probability > 0.5 else "not siren"

    print(f"\nFor file {path}:\n  Actual label: {class_label}\n  Predicted label: {prediction}")
    input("\nPress enter to continue...")

