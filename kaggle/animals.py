'''
Developer: vkyprmr
Filename: animals.py
Created on: 2020-09-20 at 14:51:30
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-21 at 00:12:43
'''

#%%
# Imports
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Callback functions
path_checkpoint = 'checkpoints/animals.keras'

""" callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True) """

""" callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=50, verbose=1) """

log_dir = 'logs/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')

callback_tensorboard = TensorBoard(log_dir=log_dir,
                                   histogram_freq=1,
                                   write_graph=False)

""" callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=25,
                                       verbose=1) """

callbacks = [callback_tensorboard]

# Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True)

#%%
# More control over Runtime device and memory
### Check compatibility with tf_v1 and v2 before running the code
"""
    Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-gm", "--gpumemory",dest = "gpu_memory", help="GPU Memory to use - default 2 GB")
    parser.add_argument("-m", "--mode",dest = "mode", help="Mode: 'static':'s' or 'dynamic':'d'")
    parser.add_argument("-bs", "--batchsize",dest = "batch_size", help="Batch size")
    parser.add_argument("-e", "--epochs",dest = "epochs", help="Epochs")
    parser.add_argument("-d", "--device",dest = "device", help="CPU/GPU - default: CPU")


    args = parser.parse_args()
    try:
        batch_size = int(args.batch_size)
    except:
        batch_size = 64
    #print(batch_size)
    try:
        mode = args.mode.lower()
    except:
        mode = 'd'
    #print(mode)
    try:
        epochs = int(args.epochs)
    except:
        epochs = 10
    #print(epochs)
    try:
        device = args.device.lower()
    except:
        device = 'no_device'
    #print(device)
    try:
        gpu_mem = int(args.gpu_memory)
    except:
        gpu_mem = 2
    #print(gpu_mem)


    if device=='gpu':
        if mode=='s' or mode=='static':
            #gpu_mem = int(args.gpu_memory)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            #The variable GB is the memory size you want to use.
            try:
                config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*gpu_mem))]
                if gpus:
                    # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
                    try:
                        tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
                        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                    except RuntimeError as e:
                        # Virtual devices must be set before GPUs have been initialized
                        print(e)
            except:
                print('Static mode selected but no memory limit set. Please set a memory limit by adding the flag -gm=X (gb) or --gpumemory=x (gb) after -m=s or --memory=s')
                quit()
        else:
            physical_devices = tf.config.experimental.list_physical_devices('GPU') 
            for physical_device in physical_devices: 
                tf.config.experimental.set_memory_growth(physical_device, True)

    else:
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
 """
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


# %%
# Building the model
model_name = f'icm-cm-cm-fdd_163x32x2-323x32x2-643x32x2-f512-3'

model = Sequential(layers=[
                            Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
                            MaxPooling2D(2,2),
                            Conv2D(32, (3,3), activation='relu'),
                            MaxPooling2D(2,2),
                            Conv2D(64, (3,3), activation='relu'),
                            MaxPooling2D(2,2),
                            Flatten(),
                            Dense(512, activation='relu'),
                            Dense(3, activation='softmax')
                          ],
                    name=model_name
                    )
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
model.summary()

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
                                                    batch_size=128,
                                                    class_mode='sparse'
                                                    )
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
                                                    'Data/cats_dogs_pandas/test/',
                                                    target_size=(128, 128),
                                                    batch_size=128,
                                                    class_mode='sparse'
                                                    )


#%%
# Training
log_dir = 'logs/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')+model_name

history = model.fit_generator(
                                train_generator,
                                steps_per_epoch=25, epochs=100,
                                verbose=1, callbacks=callbacks
                                )


# %%
# Visualizing performance
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['acc'], label='Acc')
plt.title('Loss vs. Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.legend()

# %%

img = image.load_img('Data/cats_dogs_pandas/images/cat.jpg', target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
preds = model.predict(images)
print(preds)

# %%
