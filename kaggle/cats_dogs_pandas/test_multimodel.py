'''
Developer: vkyprmr
Filename: test_multimodel.py
Created on: 2020-09-20 at 18:51:30
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-21 at 16:18:33
'''
#%%
# Imports
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib qt
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True)

from multimodels import MultiModels

#%%
### Initializing
model_params0 = {
                'ip_filters': 16,
                'kernel_size': (3,3),
                'ip_pool': (2,2),
                'range_nlayers': [32,64,128],
                'n_dense': [256],
                'op_shape': 3
                }
model_params1 = {
                'ip_filters': 16,
                'kernel_size': (3,3),
                'ip_pool': (2,2),
                'range_nlayers': [32,64,128,256],
                'n_dense': [512,256],
                'op_shape': 3
                }
""" model_params2 = {
                'ip_filters': 128,
                'kernel_size': (3,3),
                'ip_pool': (2,2),
                'range_nlayers': [32,64],
                'n_dense': [256,128],
                'op_shape': 3
                }
model_params3 = {
                'ip_filters': 16,
                'kernel_size': (3,3),
                'ip_pool': (2,2),
                'range_nlayers': [32,64,32,64],
                'n_dense': [64,32],
                'op_shape': 3
                }
model_params4 = {
                'ip_filters': 16,
                'kernel_size': (3,3),
                'ip_pool': (2,2),
                'range_nlayers': [128,256],
                'n_dense': [64],
                'op_shape': 3
                } """

model_params = [model_params0, model_params1]       #, model_params2, model_params3, model_params4

mm = MultiModels()
models = mm.build_models(model_params)

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
# Training loop

def train_model(model):
    ind_start = time.time()
    # Compiling the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001),
                metrics=['accuracy'])
    model.summary()

    # Callback functions
    ckpt_dir = 'logs/multimodels/checkpoints/'+model.name
    try:
        os.mkdir(ckpt_dir)
    except Exception as e:
        print(e)

    path_checkpoint = ckpt_dir+'/'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.ckpt'

    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_weights_only=True,
                                        save_best_only=True)


    log_dir = 'logs/multimodels/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')+'_'+model.name

    callback_tensorboard = TensorBoard(log_dir=log_dir,
                                    histogram_freq=1,
                                    write_graph=False)
    """ 
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            min_lr=1e-4,
                                            patience=5,
                                            verbose=1)
        
        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=10, verbose=1)
    """

    callbacks = [callback_tensorboard,
                callback_checkpoint]


    history = model.fit_generator(
                                    train_generator,
                                    steps_per_epoch=50, epochs=2, 
                                    validation_data=val_generator, validation_steps=50,
                                    verbose=1, callbacks=callbacks
                                    )

    models_dir = 'logs/multimodels/models/'+model.name
    try:
        os.mkdir(models_dir)
    except Exception as e:
        print(e)

    save_name = models_dir+'/'+datetime.now().strftime('%Y%m%d-%H%M%S')+'animals.h5'
    model.save(save_name)

    end_ind_time = time.time()-ind_start

    if end_ind_time>60:
        end_ind_time = end_ind_time/60
        print(f'Time taken to train {model.name} - model: {end_ind_time} hours.')
    else:
        print(f'Time taken to train {model.name}- model: {end_ind_time} minutes.')

    return history

#%%
"""
# Training
com_start = time.time()

histories = []

for m in models:
    with tf.Session() as sess:
        model = m
        h = train_model(model)
        histories.append(h)
        #tf.keras.backend.clear_session()
        sess.close()

end_time = (time.time()-com_start)/60

print(f'Training completed.')

if end_time>60:
    end_time = end_time/60
    print(f'Total time taken for training {len(models)} models: {end_time} hours.')
else:
    print(f'Total time taken for training {len(models)} models: {end_time} minutes.')
 """

#%%
# Training
for i in range(len(models)):
    model = models[i]
    print(f'Current model: {model}')
    history = train_model(model)
    i+=1

##### Difficult as shit

#%%
# Resetting for all models in a loop
from IPython.display import Javascript
Javascript('IPython.notebook.execute_cells_above()')


# %%
