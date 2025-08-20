from dotenv import load_dotenv

load_dotenv()

import sys
import time
import threading
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from queue import Queue

SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SIZE = 16000
STRIDE = 8000

print("Loading model...")
model = load_model("models\\siren_detector.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.Accuracy()]
)

def preprocess(wave):
    wav = tf.convert_to_tensor(wave, dtype=tf.float32)   # ← fix
    wav = wav[:WINDOW_SIZE]
    padding = WINDOW_SIZE - tf.shape(wav)[0]
    wav = tf.pad(wav, [[0, padding]])

    stft = tf.signal.stft(wav, frame_length=320, frame_step=32)
    mag = tf.abs(stft)

    num_mel_bins = 64
    num_spectrogram_bins = mag.shape[-1]  # 1 + frame_length//2
    lower_hz, upper_hz = 80.0, 7600.0
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SAMPLE_RATE, lower_hz, upper_hz
    )
    mel = tf.matmul(mag, linear_to_mel)
    log_mel = tf.math.log(mel + 1e-6)

    return log_mel[tf.newaxis, ..., tf.newaxis]

def process_chunk(samples, previous):
    logMel = preprocess(samples)

    yhat = model.predict(logMel, verbose=0)
    probability = float(yhat[0][0])
    prediction = True if probability > 0.5 else False

    if not prediction and not previous:
        print("\r❌No siren detected!  ", end="")
    else:
        print("\r✅Siren detected!  ", end="")

    return prediction

q = Queue()
running = True

def audio_callback(indata, frames, t, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 0].astype(np.float32, copy=True))  # mono

def worker():
    buf = np.empty(0, dtype=np.float32)
    stride = 0

    previous = False # No siren in previous clip at the beginning

    while running or not q.empty():
        chunk = q.get()
        buf = np.concatenate((buf, chunk))
        if buf.size > WINDOW_SIZE:
            buf = buf[-WINDOW_SIZE:]
        stride += chunk.size
        while stride >= STRIDE and buf.size >= WINDOW_SIZE:
            previous = process_chunk(buf[-WINDOW_SIZE:].copy(), previous)
            stride -= STRIDE
        q.task_done()

t = threading.Thread(target=worker, daemon=True)
t.start()

try:
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype='float32',
                        callback=audio_callback):
        print("recording; Ctrl+C to stop")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    running = False
    q.join()