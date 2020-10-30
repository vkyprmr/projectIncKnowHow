"""
Developer: vkyprmr
Filename: cats_dogs.py
Created on: 2020-10-6, Di., 18:58:28
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-30, Fri, 18:39:54
"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enabling dynamic GPU usage
device = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(device[0], True)
except Exception as e:
    print(f'Error: {e}')

# Data
base_dir = '../../../Data/cats_vs_dogs/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


def sample_images(directory):
    """
    Args:
        directory: pass in the directory (trein_dir or val_dir)
        The directory should be in a tree format.
        Example - train
                    - cats
                    -dogs
    Returns: random pics of cats and dogs as an image
    """
    nrows, ncols = 6, 6
    dogs = os.path.join(directory, 'dogs')
    cats = os.path.join(directory, 'cats')
    dog_files = os.listdir(dogs)
    cat_files = os.listdir(cats)
    pic_index = np.random.randint(18, len(cat_files))
    fig = plt.gcf()
    next_cat = [os.path.join(cats, cat_file) for cat_file in cat_files[pic_index - 18:pic_index]]
    next_dog = [os.path.join(dogs, dog_file) for dog_file in dog_files[pic_index - 18:pic_index]]
    for i, img_path in enumerate(next_cat + next_dog):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.tight_layout()
        plt.show()


# sample_images(val_dir)

# Building the Model
layers = [
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # Dropout(0.1),
    # Conv2D(64, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    # Conv2D(64, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    # Dropout(0.1),
    # Conv2D(32, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    # Conv2D(32, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    # Dropout(0.1),
    # Conv2D(64, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    # Conv2D(64, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    # Dropout(0.1),
    # Flatten(),
    # Dense(256, activation='relu'),
    # Dropout(0.1),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
]

model_name = f'cats_vs_dogs_{len(layers)}-layers_with_augmentation-CMCM-1D_rms-1e-3'

model = Sequential(layers=layers, name=model_name)
opt = RMSprop(lr=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Generating data from directory
train_datagen = ImageDataGenerator(rescale=1. / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )
"""
Commonly used filters:
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
Only use in train generator and never on validation generator
 """
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                    batch_size=100, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255.0)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150),
                                                batch_size=100, class_mode='categorical')

# Training
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + model_name
chkpt_dir = 'logs/saved_models/ckpt/ckpt_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + model_name + '/'
if not os.path.exists(chkpt_dir):
    os.mkdir(chkpt_dir)

path_chkpt = chkpt_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)
es_callback = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.1, verbose=1)
chkpt_callback = ModelCheckpoint(filepath=path_chkpt, monitor='val_loss',
                                 verbose=1, save_weights_only=True,
                                 save_best_only=True)
callbacks = [tb_callback, chkpt_callback, rlr_callback, es_callback]

spe = 100
vspe = 50
epochs = 100

hist = model.fit(train_generator, epochs=epochs, steps_per_epoch=spe,
                 validation_data=val_generator, validation_steps=vspe,
                 verbose=1, callbacks=callbacks)


# Plots
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
    pred_classes = []
    print(f'Found {len(imgs)} images to predict.')
    for img in tqdm(imgs, desc='Prediction progress:'):
        img_path = os.path.join(directory, img)
        pic = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(pic)
        x = x/255.0
        x = np.expand_dims(x, axis=0)
        x = np.vstack([x])
        pred = trained_model.predict(x)
        print(type(pred))
        if pred[0][0] > 0.5:
            predicted_class = 'cat'
        else:
            predicted_class = 'dog'
        # print(f'The image {img} contains a {predicted_class}. ({len(imgs)-imgs.index(img)}/{len(imgs)})')
        print(img, pred)
        preds.append(pred)
        pred_classes.append(predicted_class)

    results = pd.DataFrame(columns=['Image', 'Prediction', 'Class'])
    results.Image = imgs
    results.Prediction = preds
    results.Class = pred_classes

    return results


res = make_predictions(test_dir, model)
# image_dir = '../../../Data/cats_vs_dogs/image/single'
# res = make_predictions(image_dir, model)

# Saving the entire model
model_save_dir = 'logs/saved_models/models/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + 'model_name'
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
