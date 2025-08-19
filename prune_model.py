import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
from construct_dataset import get_datasets
import tensorflow as tf

train, test, classWeight = get_datasets()
EPOCHS = 2

T = 1 + (60000 - 320) // 32
baseModel = keras.Sequential([
    keras.layers.Input(shape=(T, 64, 1)),
      keras.layers.Conv2D(16, (3, 3), activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(96, activation='relu'),
      keras.layers.Dense(1, activation='sigmoid'),
])

# Load pretrained model weights and set parameters for pruning
baseModel.load_weights("weights\\pretrained_weights.h5")
steps_per_epoch = int(tf.data.experimental.cardinality(train).numpy())
pruning_params = {
  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
     initial_sparsity=0.0,
     final_sparsity=0.5,
     begin_step=0,
     end_step=steps_per_epoch * EPOCHS
  )
}
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

modelForPruning = tfmot.sparsity.keras.prune_low_magnitude(baseModel, **pruning_params)
del baseModel
modelForPruning.summary()
modelForPruning.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Recall(), keras.metrics.Precision()])

# Train the model
modelForPruning.fit(train, epochs=EPOCHS, validation_data=test, callbacks=callbacks)
del callbacks

# Strip the pruning
modelForExport = tfmot.sparsity.keras.strip_pruning(modelForPruning)
del modelForPruning
modelForExport.summary()
modelForExport.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Recall(), keras.metrics.Precision()])

# Evaluate the model
loss, recall, precision = modelForExport.evaluate(test)
print(f"Pruned stats:  Loss: {loss}, Precision: {precision}, Recall: {recall}")

# Save the model
modelForExport.save("models\\pruned_model.h5", include_optimizer=True)

del modelForExport, train, test