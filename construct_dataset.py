import tensorflow as tf
import os
import random
import soundfile as sf
import librosa
import tensorflow_hub as hub

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

SAMPLE_RATE = 16000
WINDOW_LENGTH = 2
WINDOW_SAMPLES = WINDOW_LENGTH * SAMPLE_RATE
TRAIN_FRACTION = 0.85
RANDOM_SEED = 1


def load_wav_16k_mono(path):
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y.astype("float32")


def slice_file(file_path, label):
    wav = tf.numpy_function(load_wav_16k_mono, [file_path], tf.float32)
    wav.set_shape([None])

    windows = tf.signal.frame(wav, frame_length=WINDOW_SAMPLES, frame_step=WINDOW_SAMPLES, pad_end=False)
    labels = tf.fill([tf.shape(windows)[0]], tf.cast(label, tf.float32))

    return tf.data.Dataset.from_tensor_slices((windows, labels))


def count_slices(path_labels):
    counts = {0: 0, 1: 0}

    for file_path, label in path_labels:
        info = sf.info(file_path)
        samples_per_window = info.samplerate * WINDOW_LENGTH
        window_count = info.frames // samples_per_window

        counts[label] += window_count

    return counts


def preprocess(wav, label):
    wav.set_shape([WINDOW_SAMPLES])
    wav = tf.clip_by_value(tf.cast(wav, tf.float32), -1.0, 1.0)

    _, embeddings, _ = yamnet(wav)

    # Describe both the overall clip and its strongest local event.
    features = tf.concat([
        tf.reduce_mean(embeddings, axis=0),
        tf.reduce_max(embeddings, axis=0),
    ], axis=0)

    features.set_shape([2048])
    return tf.stop_gradient(features), label


def build_dataset(path_labels, training):
    paths, labels = zip(*path_labels)

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    dataset = dataset.flat_map(slice_file)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_SEED, reshuffle_each_iteration=True)

    return dataset.batch(16).prefetch(tf.data.AUTOTUNE)

def get_datasets(test_fold):
    folds = [f"fold_{fold}" for fold in range(1, 11)]
    folds.remove(f"fold_{test_fold}")
    pos_paths = []
    neg_paths = []

    for fold in folds:
        pos_paths += [(f"data/{fold}/positive/{filename}", 1) for filename in os.listdir(f"data/{fold}/positive/") if filename.endswith(".wav")]
        neg_paths += [(f"data/{fold}/negative/{filename}", 0) for filename in os.listdir(f"data/{fold}/negative/") if filename.endswith(".wav")]

    train_paths = sorted(pos_paths + neg_paths)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(train_paths)

    test_paths = [(f"data/fold_{test_fold}/positive/{filename}", 1) for filename in os.listdir(f"data/fold_{test_fold}/positive/") if filename.endswith(".wav")]
    test_paths += [(f"data/fold_{test_fold}/negative/{filename}", 0) for filename in os.listdir(f"data/fold_{test_fold}/negative/") if filename.endswith(".wav")]

    train_slice_counts = count_slices(train_paths)
    test_slice_counts = count_slices(test_paths)

    negative_count = train_slice_counts[0]
    positive_count = train_slice_counts[1]
    total = negative_count + positive_count

    class_weight = {
        0: total / (2 * negative_count),
        1: total / (2 * positive_count),
    }

    train = build_dataset(train_paths, training=True)
    test = build_dataset(test_paths, training=False)

    train_length = sum(train_slice_counts.values())
    test_length = sum(test_slice_counts.values())

    print(f"Training samples: {train_length},  Test samples: {test_length},  Total samples: {train_length + test_length}")
    print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")
    print(f"Negative weight: {class_weight[0]:.3f},  Positive weight: {class_weight[1]:.3f}")

    return train, test, class_weight
