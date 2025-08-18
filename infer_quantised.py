from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import time
import numpy as np
import librosa
import tensorflow as tf

def load_wav_16k_mono(path):
    if isinstance(path, np.ndarray):
        path = path.item().decode()      # ← decode bytes→str
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y.astype("float32")

def preprocess_eager(path):
    wav = load_wav_16k_mono(path)
    wav = wav[:60000]
    wav = np.pad(wav, (0, max(0, 60000 - len(wav))))
    spec = tf.signal.stft(
        wav,
        frame_length = 512,
        frame_step = 32,
        fft_length = 512
    )
    spec = tf.abs(spec)

    T_exp = in_det["shape"][1]
    frames = tf.shape(spec)[0]
    spec = tf.cond(
        frames < T_exp,
        lambda: tf.pad(spec, [[0, T_exp - frames], [0, 0]]),
        lambda: spec[:T_exp, :]
    )

    spec = spec.numpy()
    spec = spec[..., np.newaxis]
    return spec

# Load TFLite once
model_bytes = (Path("models") / "siren_detector.tflite").read_bytes()
interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

def predict(audio_path):
    spec = preprocess_eager(audio_path)
    inp = np.expand_dims(spec, 0).astype(in_det["dtype"])
    interpreter.set_tensor(in_det["index"], inp)
    startTime = time.time()
    interpreter.invoke()
    totalTime = time.time() - startTime
    return interpreter.get_tensor(out_det["index"]), totalTime

# Usage:
while True:
    audio_path = input("Audio path: ").replace('"', '')
    result, totalTime = predict(audio_path)
    print(f"Time taken: {totalTime:.2f}\n{result}")

