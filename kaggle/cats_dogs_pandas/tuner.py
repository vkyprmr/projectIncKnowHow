'''
Developer: vkyprmr
Filename: tuner.py
Created on: 2020-09-21 at 12:49:26
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-21 at 13:39:03
'''

#%%
# Imports
import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

""" gpus = tf.config.experimental.list_physical_devices('GPU')
config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*6))]
if gpus:
    # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e) """

#%%
# Preparing data
base_dir = 'Data/cats_dogs_pandas/'
train_dir = os.path.join(base_dir, 'train')

### Individual class directories
train_dir_cats = os.path.join(train_dir, 'cats')
train_dir_dogs = os.path.join(train_dir, 'dogs')
train_dir_pandas = os.path.join(train_dir, 'pandas')

### Files
train_fnames_cats = os.listdir(train_dir_cats)
train_fnames_dogs = os.listdir(train_dir_dogs)
train_fnames_pandas = os.listdir(train_dir_pandas)

#%%
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
                                                    'Data/cats_dogs_pandas/train/',
                                                    target_size=(128, 128),
                                                    batch_size=50,
                                                    class_mode='sparse'
                                                    )
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
                                                    'Data/cats_dogs_pandas/test/',
                                                    target_size=(128, 128),
                                                    batch_size=25,
                                                    class_mode='sparse'
                                                    )


#%%
# Model

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('input_units', min_value=25, max_value=250, step=25),
                            (3,3), input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32, max_value=256, step=32), (3,3)))
        model.add(Activation('relu'))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0, max_value=0.5, step=0.1)))

    model.add(Flatten())
    for j in range(hp.Int('n_layers', 1, 2)):
        model.add(Dense(hp.Int(f'dense_{j}_units',
                                min_value=64, max_value=512, step=64), activation='relu'))
        model.add(Dropout(hp.Float(f'dropout_{j}', min_value=0, max_value=0.5, step=0.1)))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp.Choice('optimizer', ['adam','sgd','rmsprop']),
                  metrics=['accuracy'])
    return model

#%%
### Callback functions
ckpt_dir = 'logs/ktuner/checkpoints/'+datetime.now().strftime('%Y%m%d-%H%M%S')
os.mkdir(ckpt_dir)
path_checkpoint = ckpt_dir+datetime.now().strftime('%Y%m%d-%H%M%S')+'.ckpt'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=15, verbose=1)

log_dir = 'logs/ktuner/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')

callback_tensorboard = TensorBoard(log_dir=log_dir,
                                   histogram_freq=1,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=5,
                                       verbose=1)

callbacks = [callback_tensorboard,
             callback_checkpoint,
             callback_early_stopping,
             callback_reduce_lr]

klogs = f'logs/ktuner/ktuner/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.mkdir(klogs)

#%%
# RandomSearch
### Training

ktuner = RandomSearch(build_model, objective='val_acc', max_trials=3,
                      directory=klogs)

ktuner.search(
                train_generator,
                steps_per_epoch=100, epochs=100,
                validation_data=val_generator, validation_steps=10,
                verbose=1, callbacks=callbacks
                )

ktuner.results_summary()

with open(f'logs/ktuner/tuner/{datetime.now().strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as f:
    pickle.dump(ktuner, f)

#%%
# Best hyperparameters && best model
best_hp = ktuner.get_best_hyperparameters()[0].values
best_model = ktuner.get_best_models()[0].summary()
ktuner.results_summary()

#%%
# Loading the tuner if exited
tuner_file = ''
ktuner = pickle.load(open(tuner_file, 'rb'))
best_hp = ktuner.get_best_hyperparameters()[0].values
best_model = ktuner.get_best_models()[0].summary()
ktuner.results_summary()
