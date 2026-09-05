import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from dotenv import load_dotenv
import json
from tensorflow import keras
from construct_dataset import get_datasets, WINDOW_SAMPLES

load_dotenv()

EPOCHS = 50
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
        "loss": 0, "precision": 0, "recall": 0,
    } for test_fold in range(1, 11)
}


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

    loss, PR_AUC, precision_value, recall_value = model.evaluate(test)
    print(f"Baseline stats:  Loss: {loss}, PR_AUC: {PR_AUC}, Precision: {precision_value}, Recall: {recall_value}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/siren_head_test_{test_fold}.keras")

    print("Saved siren head")

    return loss, precision_value, recall_value


for test_fold in range(1, 11):
    this_loss, this_precision, this_recall = do_training(test_fold)

    model_performance[f"test_{test_fold}"]['loss'] = this_loss
    model_performance[f"test_{test_fold}"]['precision'] = this_precision
    model_performance[f"test_{test_fold}"]['recall'] = this_recall

averages = {
    metric: sum(result[metric] for result in model_performance.values()) / len(model_performance)
    for metric in ("loss", "precision", "recall")
}

print(f"Average metrics: {averages}")
with open(os.path.join("models", "model_performance.json"), "w") as file:
    json.dump({"model_performance": model_performance, "averages": averages}, file, indent=4, default=float)
