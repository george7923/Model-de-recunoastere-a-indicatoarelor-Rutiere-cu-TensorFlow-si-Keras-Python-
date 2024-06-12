import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = 'C:/Users/George/Desktop/PORTOFOLIU CALCUL/PentruChestionar/PentruTraining'

meta = pd.read_csv(os.path.join(data_path, 'Meta.csv'))
train = pd.read_csv(os.path.join(data_path, 'Train.csv'))
test = pd.read_csv(os.path.join(data_path, 'Test.csv'))

def load_images(data_frame, base_path, img_size=(32, 32)):
    images = []
    labels = []
    for index, row in data_frame.iterrows():
        img_path = os.path.join(base_path, row['Path'])
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        images.append(img)
        labels.append(row['ClassId'])
    return np.array(images), np.array(labels)

X_train, y_train = load_images(train, data_path)
X_test, y_test = load_images(test, data_path)

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, num_classes=np.max(y_train)+1)
y_test = to_categorical(y_test, num_classes=np.max(y_test)+1)

model_path = os.path.join(data_path, 'PentruChestionare2.keras')
model = load_model(model_path)

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

model.save(model_path)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
