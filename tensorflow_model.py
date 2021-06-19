# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:43:33 2021

@author: Peter L'Oiseau
"""

#import neccesary libraries to build a classification model
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

# working directory
path = 'C:/Users/peter/Documents/baseball-databases/savant_video'
os.chdir(path)

# the dataset file or root folder path
dataset_path = 'C:/Users/peter/Documents/baseball-databases/savant_video/pics' 
#standardize height and width of the pictures
IMG_HEIGHT = 64 
IMG_WIDTH = 64 
#set a batch size for training
batch_size = 32

#define the training and validation sets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="training",
  seed=12,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="validation",
  seed=12,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batch_size)

#ensure the classes have been imported
class_names = train_ds.class_names
print(class_names)

#visualize a sample of the pics to ensure they have loaded
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batch_size)

#keep the photos in cache to speed of modelling process
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#normalize the photos to a specific size
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#change pixel values to a 0-1 scale
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

#augment the data to create more data to build the model
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(IMG_HEIGHT, 
                                                              IMG_WIDTH,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

params = dict(epochs = [100,200], dropout=[.5,.35,.2,.1], 
              learning_rate=[.001,.005,.01], kernel=[2,3,4],
              strides=[1,2,3], dilate=[1,2,3])    

num_classes = 4

def get_new_model(lr, dropout, kernel, strides, dilate):
     #initialize the model structure for a CNN
    model = Sequential([
      data_augmentation,
      layers.experimental.preprocessing.Rescaling(1./255),
      layers.Conv2D(32, (kernel,kernel), padding='same', activation='relu', strides=(strides,strides), dilation_rate=(dilate,dilate)),
      #layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(dropout),
      layers.Conv2D(64, (kernel,kernel), padding='same', activation='relu', strides=(strides,strides), dilation_rate=(dilate,dilate)),
      #layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(dropout),
      layers.Conv2D(128, (kernel,kernel), padding='same', activation='relu', strides=(strides,strides), dilation_rate=(dilate,dilate)),
      #layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(dropout),
      layers.Conv2D(256, (kernel,kernel), padding='same', activation='relu', strides=(strides,strides), dilation_rate=(dilate,dilate)),
      #layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(dropout),
      layers.Conv2D(512, (kernel,kernel), padding='same', activation='relu', strides=(strides,strides), dilation_rate=(dilate,dilate)),
      #layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(dropout),
      layers.Conv2D(1024, (kernel,kernel), padding='same', activation='relu', strides=(strides,strides), dilation_rate=(dilate,dilate)),
      #layers.BatchNormalization(),
      layers.MaxPooling2D(),
      layers.Dropout(dropout),
      layers.Flatten(),
      layers.Dense(2048, activation='relu'),
      layers.Dropout(dropout),
      layers.Dense(4096, activation='relu'),
      #layers.BatchNormalization(),
      layers.Dropout(dropout),
      layers.Dense(num_classes, activation='softmax')
    ])
    
    #compile the model    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


#instantiate a random search where 20 of the combos are tried
combos = np.prod([len(x) for x in list(params.values())])
rand = random.sample(range(combos),100)
    
k=0
for kernel in  list(params.values())[3]:                
    for strides in list(params.values())[4]:        
        for dilate in list(params.values())[5]:
            for epochs in list(params.values())[0]:
                for dropout in list(params.values())[1]:
                    for lr in  list(params.values())[2]:    
                     if k in rand:
                        #if passes random search, build and comile the model
                        try:
                            model=get_new_model(lr, dropout, kernel, strides, dilate)
                            #fit the model
                            history=model.fit(
                              train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks = EarlyStopping(monitor = 'val_loss', patience = 20))
                            
                            val_acc = max(history.history['val_accuracy'])
                            par_set = ['adam' , epochs, dropout, lr, kernel, strides, dilate, val_acc]
                            row = pd.DataFrame([par_set])
                            row.to_csv('hyperpar.csv', mode='a', header=False, index=False)                       
                        except:
                            pass
                     else:
                        pass
                     k+=1
                     print(k)

#choose best model
hyper_par=pd.read_csv('hyperpar.csv')
optimizer, epochs, dropout, lr, kernel, strides, dilate, acc = hyper_par.loc[hyper_par['val_acc'].idxmax(), ]

#and build it
model = get_new_model(lr, dropout, kernel, strides, dilate)

wname = 'weights_0.hdf5'
#make sure to save the best weights
checkpoint = ModelCheckpoint(wname, monitor='val_accuracy',
                 save_best_only = True)

callbacks_list = [EarlyStopping(monitor = 'val_loss', patience = 50),
          checkpoint]

#fit the model
history=model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs*5,
  callbacks = callbacks_list)

model.load_weights(wname)

#visualize the training results to see where we may improve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
    
#predict test set and view confusion matrix
predictions = model.predict(val_ds)
predicted_categories = tf.argmax(predictions, axis=1)
true_categories = tf.concat([y for x, y in val_ds], axis=0)

print(tf.math.confusion_matrix(true_categories, predicted_categories))
print(class_names)
print(model.evaluate(val_ds))

#save the model
#!mkdir -p saved_model
model.save('keras_model.h5')
