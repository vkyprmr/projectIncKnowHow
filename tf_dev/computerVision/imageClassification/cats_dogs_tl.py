"""
Developer: vkyprmr
Filename: cats_dogs_tl.py
Created on: 2020-10-8, Do., 15:37:37
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-8, Do., 16:29:53
"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

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

# Pre-trained model
weights_file = 'transfer_learning/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape=(128, 128, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Building the model
'''
    First flatten the output from the last layer and then add DNN
 '''
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)

model_name = 'cats_vs_dogs-transfer_learning-inception_v3'
model = Model(pre_trained_model.input, x, name=model_name)
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
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

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128),
                                                    batch_size=100, class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1. / 255.0)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(128, 128),
                                                batch_size=100, class_mode='binary')

# Training
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_name
chkpt_dir = 'logs/checkpoints' + model_name + '/'
if not os.path.exists(chkpt_dir):
    os.mkdir(chkpt_dir)

path_chkpt = chkpt_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)
es_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, verbose=1)
chkpt_callback = ModelCheckpoint(filepath=path_chkpt, monitor='val_loss',
                                 verbose=1, save_weights_only=True,
                                 save_best_only=True)
callbacks = [tb_callback, es_callback, rlr_callback, chkpt_callback]

spe = 100
vspe = 50
epochs = 100

history = model.fit(train_generator, epochs=epochs, steps_per_epoch=spe,
                    validation_data=val_generator, validation_steps=vspe,
                    verbose=1, callbacks=callbacks)


# Plots
def plot_metrics():
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


# plot_metrics()


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
    for img in imgs:
        img_path = os.path.join(test_dir, img)
        pic = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(pic)
        x = np.expand_dims(x, axis=0)
        x = np.vstack([x])
        pred = trained_model.predict(x)
        if pred > 0.5:
            predicted_class = 'cat'
        else:
            predicted_class = 'dog'
        print(f'The image {img} contains a {predicted_class}. ({len(imgs) - imgs.index(img)}/{len(imgs)})')
        preds.append(pred)
        pred_classes.append(predicted_class)

    results = pd.DataFrame(columns=['Image', 'Prediction', 'Class'])
    results.Image = imgs
    results.Prediction = preds
    results.Class = pred_classes

    return results

# res = make_predictions(test_dir, model)


# model.save('logs/model/cats_vs_dogs/model')

# tf.keras.models.load_model('logs/model/cats_vs_dogs/')
