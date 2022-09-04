"""
Developer: vkyprmr
Filename: cdp.py
Created on: 2020-09-20 at 14:51:30
"""
"""
Modified by: vkyprmr
Last modified on: 2020-11-1, Sun, 15:8:45
"""

# Imports
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from tqdm import tqdm
#%matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

# Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True)

# Preparing data
base_dir = '../../../Data/cats_dogs_pandas/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

### Individual class directories
train_dir_cats = os.path.join(train_dir, 'cats')
train_dir_dogs = os.path.join(train_dir, 'dogs')
train_dir_pandas = os.path.join(train_dir, 'panda')

### Files
train_fnames_cats = os.listdir(train_dir_cats)
train_fnames_dogs = os.listdir(train_dir_dogs)
train_fnames_pandas = os.listdir(train_dir_pandas)


""" 
# Looking at some images
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 6

pic_index = 0 # Index for iterating over images

#%%
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pic = [os.path.join(train_dir_cats, fname)
                for fname in train_fnames_cats[ pic_index-8:pic_index]
               ]

next_dog_pic = [os.path.join(train_dir_dogs, fname)
                for fname in train_fnames_dogs[ pic_index-8:pic_index]
               ]

next_panda_pic = [os.path.join(train_dir_pandas, fname)
                for fname in train_fnames_pandas[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pic+next_dog_pic+next_panda_pic):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)
    # plt.tight_layout()

 """
# Genearting data using ImageDataGenerator
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )
train_generator = train_datagen.flow_from_directory(
                                                    train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='sparse'
                                                    )
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
                                                    val_dir,
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='sparse'
                                                    )


# Building the model
layers = [
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3), padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D(2, 2),
    Dropout(0.1),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    # MaxPooling2D(2, 2),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.1),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.1),
    # Conv2D(512, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D(2, 2),
    # Conv2D(512, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D(2, 2),
    # Dropout(0.1),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
]

model_name = f'cdp_{len(layers)}-layersWaug-CMCMD_BN-1D_sgd1e-3'

model = Sequential(layers=layers, name=model_name)
opt = SGD(lr=1e-3, momentum=0.9)      # , momentum=0.9
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Training
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + model_name
chkpt_dir = 'logs/ckpt/cdp/'

if not os.path.exists(chkpt_dir):
    print('Created ckpt directory.')
    os.mkdir(chkpt_dir)

path_chkpt = chkpt_dir + datetime.now().strftime('%Y%m%d-%H%M%S') + '.ckpt'

path_chkpt = Path(path_chkpt)

tb_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)
es_callback = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.1, verbose=1)
chkpt_callback = ModelCheckpoint(filepath=path_chkpt.absolute(), monitor='val_loss',
                                 verbose=1, save_weights_only=True,
                                 save_best_only=True)
callbacks = [tb_callback, chkpt_callback, rlr_callback, es_callback]

spe = 25
vspe = 25
epochs = 100

hist = model.fit(train_generator, epochs=epochs, # steps_per_epoch=spe,
                 validation_data=val_generator, # validation_steps=vspe,
                 verbose=1, callbacks=callbacks)


# Visualizing Learning Rates
def plot_metrics(history):
    """
    Args:
        history: history object assigned while training

    Returns:

    """
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all')
    ax[0].plot(history.history['accuracy'], label='train_acc')
    ax[0].plot(history.history['val_accuracy'], label='val_acc')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Train vs. Validation Accuracy')
    ax[1].plot(history.history['loss'], label='train_loss')
    ax[1].plot(history.history['val_loss'], label='val_loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Train vs. Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.show()


# plot_metrics(hist)


def make_predictions(directory, trained_model):
    """
    Args:
        directory: directory where test images are located
        trained_model: trained model
    Returns: a dataframe containing names of images and respective predictions and classes
    """
    imgs = os.listdir(directory)
    preds = []
    # pred_classes = []
    print(f'Found {len(imgs)} images to predict.')
    for img in tqdm(imgs, desc='Prediction progress:'):
        img_path = os.path.join(directory, img)
        pic = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(pic)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = np.vstack([x])
        pred = trained_model.predict(x)
        # print(type(pred))
        # print(pred)
        # print(pred.shape)
        # if pred[0][0] > 0.5:
        #     predicted_class = 'dog'
        #     preds.append(pred[0][0]*100)
        # else:
        #     predicted_class = 'cat'
        #     preds.append((1 - pred[0][0]) * 100)
        # print(f'The image {img} contains a {predicted_class}. ({len(imgs)-imgs.index(img)}/{len(imgs)})')
        # print(img, pred)
        preds.append(pred)
        # pred_classes.append(predicted_class)

    results = pd.DataFrame(columns=['Image', 'Predicted_class', 'Prediction_prob'])
    results.Image = imgs
    results.Prediction_prob = preds
    # results.Predicted_class = pred_classes

    return results


# res = make_predictions(test_dir, model)
# image_dir = '../../../Data/cats_vs_dogs/sample_test/'
# res = make_predictions(image_dir, model)


# Saving the entire model
model_save_dir = 'logs/saved_models/' + datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(model_save_dir)

# Loading the model
# tf.keras.models.load_model(model_save_dir)

# Loading weights
# ckpt_dir = 'logs/checkpointscats_vs_dogs_21-layers/'
# ckpt = tf.train.latest_checkpoint(ckpt_dir)
#
# model.load_weights(ckpt)
#
# loss, acc = model.evaluate_generator(val_generator, verbose=1)
# print(f'Loss: {loss}, Accuracy: {acc}')
