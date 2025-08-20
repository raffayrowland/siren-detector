import tensorflow as tf
import librosa

def load_wav_16k_mono(path):
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

def get_datasets():
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
        0: total / (2 * negativeCount) * 2.5,  # Penalise false positives
        1: total / (2 * positiveCount),
    }

    print(f"Negative weight: {classWeight[0]},  Positive weight: {classWeight[1]}")

    data = positives.concatenate(negatives) # concatenate them to one dataset

    data = data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.shuffle(buffer_size=1000)
    data = data.batch(16)
    data = data.prefetch(8)

    # define the training and testing partitions
    train = data.take(int(len(data) * 0.7))
    test = data.skip(int(len(data) * 0.7)).take(len(data) - int(len(data) * 0.7))

    return train, test, classWeight