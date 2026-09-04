from dotenv import load_dotenv
import json
import os
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
        keras.layers.Input(shape=(T, 64, 1)),

        keras.layers.Conv2D(
            16,
            kernel_size=(5, 5),
            strides=(4, 2),
            padding="same",
            use_bias=False,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),

        keras.layers.SeparableConv2D(
            24,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.SpatialDropout2D(0.1),

        keras.layers.SeparableConv2D(
            48,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 2)),
        keras.layers.SpatialDropout2D(0.1),

        keras.layers.AveragePooling2D(pool_size=(1, 8)),
        keras.layers.Reshape((-1, 48)),
        keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True)),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[recall, precision])
    model.summary()

    # Train the model
    hist = model.fit(train, epochs=EPOCHS, validation_data=test, class_weight=class_weight, callbacks=[earlyStop])

    loss, recall_value, precision_value = model.evaluate(test)
    print(f"Baseline stats:  Loss: {loss}, Precision: {precision_value}, Recall: {recall_value}")

    # Save model
    os.makedirs("models", exist_ok=True)

    try:
        model.export(f"models/siren_detector_test_{test_fold}")
        print("Saved model in keras format")

    except Exception as e:
        print(e)
        model.save(os.path.join(f"models/siren_detector_test_{test_fold}.h5"))
        print("Saved model")

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
