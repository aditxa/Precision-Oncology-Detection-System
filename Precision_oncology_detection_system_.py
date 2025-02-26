# -*- coding: utf-8 -*-
"""Precision Oncology Detection System (Tumor Imaging)

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dcY1XTFWokMxsXuy4x_2Rnwnn-4KyFUf
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

! kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip

def get_class_paths(path):
  classes = []
  classpaths = []

  for lable in os.listdir(path):
    lable_path = os.path.join(path ,lable)

    if os.path.isdir(lable_path):

      for image in os.listdir(lable_path):
        image_path = os.path.join(lable_path , image)

        classes.append(lable)
        classpaths.append(image_path)

  df = pd.DataFrame({
      'Class Path' : classpaths,
      'Class' : classes
  })
  return df

tr_df = get_class_paths('/content/Training')
class_count=tr_df['Class'].value_counts()
class_count

tr_df

# from google.colab import drive
# drive.mount('/content/drive')

ts_df = get_class_paths('/content/Testing')
class_count=ts_df['Class'].value_counts()
class_count

ts_df

plt.figure(figsize = (15,7))
ax = sns.countplot(data = tr_df , x = tr_df['Class'])

plt.figure(figsize = (5,10))
ax = sns.countplot(data = ts_df , x = ts_df['Class'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten , Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision , Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

valid_df , ts_df = train_test_split(ts_df , train_size=0.5 , stratify=ts_df['Class'])

batch_size = 32

image_size = (299,299)

image_generator = ImageDataGenerator(rescale = 1/255 , brightness_range = (0.8,1.2))

ts_generator = ImageDataGenerator(rescale = 1/255 )

tr_gen = image_generator.flow_from_dataframe(
    tr_df , x_col = 'Class Path',
    y_col = 'Class', batch_size = batch_size,
    target_size = image_size
)

valid_gen = image_generator.flow_from_dataframe(
    valid_df , x_col = 'Class Path',
    y_col = 'Class', batch_size = batch_size,
    target_size = image_size
)

ts_gen = ts_generator.flow_from_dataframe(
    ts_df , x_col = 'Class Path',
    y_col = 'Class', batch_size = 16,
    target_size = image_size, shuffle = False
)

plt.figure(figsize=(20,20))
for i in range (16):
  plt.subplot(4,4,i+1)
  batch = next(tr_gen)
  image = batch[0][0]
  lable =batch[1][0]
  plt.imshow(image)

  class_index = np.argmax(lable)

  class_names = list(tr_gen.class_indices.keys())
  class_indices = list(tr_gen.class_indices.values())

  index_position = class_indices.index(class_index)

  class_name = class_names[index_position]

  plt.title(f'Class: {class_name}')
  plt.axis('off')

plt.tight_layout()
plt.show

img_shape = (299, 299, 3)

base_model = tf.keras.applications.Xception(
    include_top = False,
    weights = 'imagenet',
    input_shape = img_shape,
    pooling = 'max'
)

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate = 0.3),      # used to prevent overfitting
    Dense(128, activation = 'relu'),
    Dropout(rate = 0.25),
    Dense(4, activation = 'softmax')
])

model.compile(Adamax(learning_rate= 0.001),
              loss = 'categorical_crossentropy',
              metrics = [
                  'accuracy',
                  Precision(),
                  Recall()
              ])

hist = model.fit(
    tr_gen,
    epochs = 30,
    validation_data = valid_gen,)

metrics = ['accuracy' , 'loss' , 'precision' , 'recall']
tr_metrics = {m: hist.history[m] for m in metrics}
val_metrics = {m: hist.history[f'val_{m}'] for m in metrics}

best_epochs = {}
best_values = {}

for m in metrics:
    if m == 'loss' :
      idx = np.argmin(val_metrics[m])
    else:
      idx = np.argmax(val_metrics[m])
    best_epochs[m] = idx+1
    best_values[m] = val_metrics[m][idx]

plt.figure(figsize = (20,12))
plt.style.use('fivethirtyeight')

for i, metric in enumerate(metrics,1):
    plt.subplot(2,2,i)
    epochs = range(1,len(tr_metrics[metric]) +1 )

    plt.plot(epochs , tr_metrics[metric], 'r', label = f'Training {metric}')
    plt.plot(epochs , val_metrics[metric], 'g', label = f'Validation {metric}')
    plt.scatter(best_epochs[metric], best_values[metric], s=150 , c = 'b',
              label = f'Best Epoch {best_epochs[metric]}')

    plt.title(f'Training and Validation {metric.title()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid(True)

plt.suptitle('Model Training Metrices over Epochs', fontsize = 16)
plt.show()

train_score = model.evaluate(tr_gen , verbose = 1)
valid_score = model.evaluate(valid_gen , verbose = 1)
test_score = model.evaluate(ts_gen , verbose = 1)

print(f'Train Accuracy: {train_score[1]*100:.2f}%')
print(f'Train Loss: {train_score[0]:.4f}')
print(f'\n\nValidation Accuracy: {valid_score[1]*100:.2f}%')
print(f'Validation Loss: {valid_score[0]:.4f}')
print(f'\n\nTest Accuracy: {test_score[1]*100:.2f}%')
print(f'Test Loss: {test_score[0]:.4f}')

# broo yaha pe confusion matrix chaiye
preds = model.predict(ts_gen)
y_pred = np.argmax(preds , axis = 1)

class_dict = {
    0: 'glioma',
    1: 'meningioma',
    2: 'no_tumor',
    3: 'pituatary'
}

cm = confusion_matrix(ts_gen.classes , y_pred)
labels = list(class_dict.keys())
plt.figure(figsize = (10,8))
sns.heatmap(cm , annot = True, fmt = 'd' , cmap = 'Blues', xticklabels = labels , yticklabels = labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from PIL import Image

def predict(img_path: str) -> None:

  #get class labels
  labels = list(class_dict.keys())

  plt.figure(figsize=(6,8))

  #load and preprocess image
  img = Image.open(img_path)
  resized_img = img.resize((299,299))
  img_array = np.asarray(resized_img)
  img_array = np.expand_dims(img_array , axis = 0) /255.0

  #get model predictions
  predictions = model.predict(img_array)
  probabilities = list(predictions[0])

  #Get predicted class
  predicted_class_idx = np.argmax(probabilities)
  predicted_class = class_dict[ predicted_class_idx]

  #plot original image
  plt.subplot(2,1,1)
  plt.imshow(resized_img)
  plt.title(f'Input MRI Image/nPredicted: {predicted_class}')

  #plot prediction probabilites
  plt.subplot(2,1,2)
  bars = plt.barh(labels , probabilities)
  plt.xlabel('Probabilites', fontsize = 15)
  plt.title('Class Probabilities')

  #add probability labels to bar
  ax = plt.gca()
  ax.bar_label(bars , fmt='%.2f')

  plt.tight_layout()
  plt.show()

  print(f'Predicted Tumour type: {predicted_class}')

predict("/content/Testing/pituitary/Te-pi_0011.jpg")

model.save_weights("xception_model.weights.h5")

from tensorflow.keras.layers import Conv2D , MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

# Create a Sequential model
cnn_model = Sequential()

# Convolution layers with Batch Normalization
cnn_model.add(Conv2D(512, (3, 3), padding='same', input_shape=(299, 299, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

cnn_model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output for fully connected layers
cnn_model.add(Flatten())

# Fully connected layers
cnn_model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
cnn_model.add(Dropout(0.35))

# Output layer
cnn_model.add(Dense(4, activation='softmax'))

# Compile the model
cnn_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics =['accuracy', Precision(), Recall()])

cnn_model.summary()

# history = cnn_model.fit(
#     tr_gen ,
#     epochs = 10 ,
#     validation_data = valid_gen,)
hist = model.fit(
    tr_gen,
    epochs = 5,
    validation_data = valid_gen,)

train_score = cnn_model.evaluate(tr_gen , verbose = 1)
valid_score = cnn_model.evaluate(valid_gen , verbose = 1)
test_score = cnn_model.evaluate(ts_gen , verbose = 1)

print(f'Train Accuracy: {train_score[1]*100:.2f}%')
print(f'Train Loss: {train_score[0]:.4f}')
print(f'\n\nValidation Accuracy: {valid_score[1]*100:.2f}%')
print(f'Validation Loss: {valid_score[0]:.4f}')
print(f'\n\nTest Accuracy: {test_score[1]*100:.2f}%')
print(f'Test Loss: {test_score[0]:.4f}')