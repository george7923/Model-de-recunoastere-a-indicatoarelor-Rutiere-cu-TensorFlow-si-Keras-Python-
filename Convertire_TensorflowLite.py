import tensorflow as tf


keras_model_path = 'C:\\Users\\George\\Desktop\\PORTOFOLIU CALCUL\\PentruChestionar\\PentruTraining\\PentruChestionare7.keras'
model = tf.keras.models.load_model(keras_model_path)

saved_keras_model_path = 'C:\\Users\\George\\Desktop\\PORTOFOLIU CALCUL\\PentruChestionar\\PentruTraining\\PentruChestionare7.keras'
model.save(saved_keras_model_path)

saved_model_path = 'C:\\Users\\George\\Desktop\\PORTOFOLIU CALCUL\\PentruChestionar\\PentruTraining\\SavedModel'
model.export(saved_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_model_path = 'C:\\Users\\George\\Desktop\\PORTOFOLIU CALCUL\\PentruChestionar\\PentruTraining\\PentruChestionare7.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'Modelul TensorFlow Lite a fost salvat la: {tflite_model_path}')
