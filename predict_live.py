import os
import sys
import time
import threading
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
import sounddevice as sd
import tensorflow as tf

SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SIZE = 32000
BLOCK_SIZE = 512
PREDICTION_INTERVAL = 0.05
SIREN_THRESHOLD = 0.5
REQUIRED_SIREN_PREDICTIONS = 10
MODEL_PATH = "models/siren_head_test_9.keras"


print("Loading models...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No classifier found at {MODEL_PATH}; run train.py first")

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = keras.models.load_model(MODEL_PATH)


@tf.function(input_signature=[tf.TensorSpec([WINDOW_SIZE], tf.float32)])
def predict(samples):
    samples = tf.clip_by_value(samples, -1.0, 1.0)
    _, embeddings, _ = yamnet(samples)

    features = tf.concat([
        tf.reduce_mean(embeddings, axis=0),
        tf.reduce_max(embeddings, axis=0),
    ], axis=0)

    return classifier(features[tf.newaxis, :], training=False)[0, 0]


audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
buffer_lock = threading.Lock()
buffer_ready = threading.Event()
samples_recorded = 0
running = True


def audio_callback(indata, frames, _time_info, status):
    global samples_recorded

    if status:
        print(status, file=sys.stderr)

    samples = indata[:, 0]
    with buffer_lock:
        audio_buffer[:-frames] = audio_buffer[frames:]
        audio_buffer[-frames:] = samples
        samples_recorded = min(WINDOW_SIZE, samples_recorded + frames)
        if samples_recorded == WINDOW_SIZE:
            buffer_ready.set()


def inference_worker():
    consecutive_sirens = 0
    buffer_ready.wait()

    while running:
        started = time.perf_counter()

        with buffer_lock:
            samples = audio_buffer.copy()

        probability = float(predict(samples).numpy())
        if probability >= SIREN_THRESHOLD:
            consecutive_sirens += 1
        else:
            consecutive_sirens = 0

        if consecutive_sirens >= REQUIRED_SIREN_PREDICTIONS:
            message = "Siren detected!"
        else:
            message = f"Listening... {consecutive_sirens}/{REQUIRED_SIREN_PREDICTIONS}"

        print(f"\r{message:<24}", end="", flush=True)

        elapsed = time.perf_counter() - started
        time.sleep(max(0, PREDICTION_INTERVAL - elapsed))


# Build the TensorFlow graph before recording starts.
predict(np.zeros(WINDOW_SIZE, dtype=np.float32)).numpy()

worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        latency="low",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        print("Recording; Ctrl+C to stop")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    running = False
    buffer_ready.set()
    worker.join()
    print()
