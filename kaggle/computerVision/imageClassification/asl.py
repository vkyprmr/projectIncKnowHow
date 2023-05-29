"""
Developer: vkyprmr
Filename: asl.py
Created on: 2020-09-22 at 21:01:50
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-22 at 23:04:53
"""

# Imports
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, \
    LearningRateScheduler

# Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# Preparing data
base_dir = '../Data/asl/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Genearting data using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=50,
    class_mode='sparse'
)
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=25,
    class_mode='sparse'
)

# Building the model
model_name = 'IC150M_2CMD-128256_2D-512256_RMSProp'

model = Sequential(layers=[
    Conv2D(150, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.15),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dense(29, activation='softmax')
],
    name=model_name
)
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
model.summary()

# Training
# Callback functions

# Training
epochs = 50
lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 10))
history_lrs = model.fit_generator(
    train_generator,
    steps_per_epoch=150, epochs=100,
    validation_data=val_generator, validation_steps=75,
    verbose=1, callbacks=[lr_schedule]
)
# Visualizing Learning Rates
plt.semilogx(history_lrs.history['lr'], history_lrs.history["loss"])
# plt.axis([1e-8, 1e-3, 0, 300])


# Actual Training
log_dir = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '__' + model_name

ckpt_dir = 'logs/checkpoints/' + model_name + '/'
os.mkdir(ckpt_dir)
path_checkpoint = ckpt_dir + datetime.now().strftime('%Y%m%d-%H%M%S') + '.ckpt'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=25, verbose=1)

log_dir = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')

callback_tensorboard = TensorBoard(log_dir=log_dir,
                                   histogram_freq=1,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=10,
                                       verbose=1)

callbacks = [callback_tensorboard,
             callback_checkpoint,
             callback_early_stopping,
             callback_reduce_lr]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=25, epochs=100,
    validation_data=val_generator, validation_steps=10,
    verbose=1, callbacks=callbacks
)

models_dir = 'logs/models/' + model_name + '/'
os.mkdir(models_dir)
save_name = models_dir + datetime.now().strftime('%Y%m%d-%H%M%S') + 'animals.h5'
model.save(save_name)

# Visualizing performance
plt.figure()
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.title('Train vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Train vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

img = image.load_img('../Data/cats_dogs_pandas/images/panda.jpg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
preds = model.predict(images)
print(preds)
