import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from dotenv import load_dotenv
import json
import numpy as np
from tensorflow import keras
from construct_dataset import get_datasets, WINDOW_SAMPLES

load_dotenv()

EPOCHS = 50
MIN_RECALL = 0.9
THRESHOLDS = tuple(step / 100 for step in range(1, 100))
FRAME_LENGTH = 320
FRAME_STEP = 32
T = 1 + (WINDOW_SAMPLES - FRAME_LENGTH) // FRAME_STEP

precision = keras.metrics.Precision(name='precision')
recall = keras.metrics.Recall(name='recall')
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

beta = 0.5
betaSquared = beta**2

model_performance = {
    f"test_{test_fold}": {
        "loss": 0, "threshold": 0, "precision": 0, "recall": 0,
    } for test_fold in range(1, 11)
}


def select_threshold(labels, probabilities):
    best_result = None

    for threshold in THRESHOLDS:
        predictions = probabilities >= threshold
        true_positives = np.sum(predictions & (labels == 1))
        false_positives = np.sum(predictions & (labels == 0))
        false_negatives = np.sum(~predictions & (labels == 1))

        precision_value = true_positives / max(true_positives + false_positives, 1)
        recall_value = true_positives / max(true_positives + false_negatives, 1)

        if recall_value <= MIN_RECALL:
            continue

        result = {
            "threshold": float(threshold),
            "precision": float(precision_value),
            "recall": float(recall_value),
        }

        if best_result is None or (
            result["precision"], result["recall"], result["threshold"]
        ) > (
            best_result["precision"], best_result["recall"], best_result["threshold"]
        ):
            best_result = result

    if best_result is None:
        raise ValueError(
            f"No threshold from 0.01 to 0.99 produced recall above {MIN_RECALL:.2f}"
        )

    return best_result


def do_training(test_fold):
    train, test, class_weight = get_datasets(test_fold)

    model = keras.Sequential([
        keras.layers.Input(shape=(2048,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.03),
        metrics=[
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    # Train the model
    hist = model.fit(train, epochs=EPOCHS, validation_data=test, class_weight=class_weight, callbacks=[earlyStop])

    evaluation = model.evaluate(test, return_dict=True)
    labels = np.concatenate([
        batch_labels.numpy().reshape(-1) for _, batch_labels in test
    ])
    probabilities = model.predict(test, verbose=0).reshape(-1)
    selected = select_threshold(labels, probabilities)

    print(
        f"Selected stats: Loss: {evaluation['loss']}, "
        f"PR_AUC: {evaluation['pr_auc']}, Threshold: {selected['threshold']:.2f}, "
        f"Precision: {selected['precision']}, Recall: {selected['recall']}"
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/siren_head_test_{test_fold}.keras")

    print("Saved siren head")

    return evaluation["loss"], selected


for test_fold in range(1, 11):
    this_loss, selected = do_training(test_fold)

    model_performance[f"test_{test_fold}"]['loss'] = this_loss
    model_performance[f"test_{test_fold}"]['threshold'] = selected['threshold']
    model_performance[f"test_{test_fold}"]['precision'] = selected['precision']
    model_performance[f"test_{test_fold}"]['recall'] = selected['recall']

averages = {
    metric: sum(result[metric] for result in model_performance.values()) / len(model_performance)
    for metric in ("loss", "precision", "recall")
}

print(f"Average metrics: {averages}")
with open(os.path.join("models", "model_performance.json"), "w") as file:
    json.dump({"model_performance": model_performance, "averages": averages}, file, indent=4, default=float)
