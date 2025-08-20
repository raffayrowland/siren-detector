from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import time
import numpy as np
import librosa
import tensorflow as tf

def load_wav_16k_mono(path):
    if isinstance(path, np.ndarray):
        path = path.item().decode()
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype("float32")


def preprocess_eager(path):
    wav = load_wav_16k_mono(path)
    wav = wav[:60000]
    wav = np.pad(wav, (0, max(0, 16000 - len(wav))))

    spec = tf.signal.stft(
        wav,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )
    mag = tf.abs(spec)

    mel = tf.matmul(mag, mel_matrix)

    log_mel = tf.math.log(mel + 1e-6).numpy()[..., np.newaxis]

    T_exp = in_det["shape"][1]
    frames = log_mel.shape[0]
    if frames < T_exp:
        pad_amt = T_exp - frames
        log_mel = np.pad(log_mel, ((0, pad_amt), (0, 0), (0, 0)))
    else:
        log_mel = log_mel[:T_exp, ...]

    return log_mel

model_bytes = (Path("models") / "siren_detector.tflite").read_bytes()
interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

sample_rate = 16000
frame_length = 320
frame_step   = 32
fft_length   = 512
num_mel_bins = in_det["shape"][2]
num_spec_bins= fft_length // 2 + 1
lower_hz, upper_hz = 80.0, 7600.0

mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins,
    num_spec_bins,
    sample_rate,
    lower_hz,
    upper_hz
)
def predict(audio_path):
    spec = preprocess_eager(audio_path)
    inp = np.expand_dims(spec, 0).astype(in_det["dtype"])
    interpreter.set_tensor(in_det["index"], inp)
    startTime = time.time()
    interpreter.invoke()
    totalTime = time.time() - startTime
    return interpreter.get_tensor(out_det["index"]), totalTime


# Usage loop unchanged
while True:
    audio_path = input("Audio path: ").replace('"', '')
    result, totalTime = predict(audio_path)
    print(f"Time taken: {totalTime:.2f}\n{result}")
