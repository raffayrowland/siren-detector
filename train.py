from dotenv import load_dotenv

load_dotenv()

# ----- PROCESS DATA ------

from construct_dataset import get_datasets

train, test, classWeight = get_datasets()

train = train.cache()
test = test.cache()

# ----- TRAIN -----

from tensorflow import keras

EPOCHS = 1
T = 1 + (16000 - 320) // 32

# define the model's layers
model = keras.Sequential([
  keras.layers.Input(shape=(T, 64, 1)),
  keras.layers.Conv2D(16, (3, 3), activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(96, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Recall(), keras.metrics.Precision()])
model.summary()


# Train the model
hist = model.fit(train, epochs=EPOCHS, validation_data=test, class_weight=classWeight)

loss, recall, precision = model.evaluate(test)
print(f"Baseline stats:  Loss: {loss}, Precision: {precision}, Recall: {recall}")

# ----- SAVE MODEL -----

model.save_weights("weights\\pretrained.weights.h5")

try:
    model.export("models\\siren_detector")
    model.save("models\\siren_detector.h5")
    print("Saved model in keras format")

except Exception as e:
    print(e)
    model.save("models\\siren_detector.h5")
    print("Saved model")
