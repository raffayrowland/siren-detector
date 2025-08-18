import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("models\\siren_detector")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tfLiteQuantModel = converter.convert()

with open("models\\siren_detector.tflite", "wb") as f:
    f.write(tfLiteQuantModel)