'''
Developer: vkyprmr
Filename: comparing_performance.py
Created on: 2020-09-24 at 15:20:21
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-25 at 12:05:41
'''

#%%
from models import Models
import os
import time
import glob
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib qt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

""" # Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True) """

gpu_mem = 6
gpus = tf.config.experimental.list_physical_devices('GPU')
#The variable GB is the memory size you want to use.
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

#%%
# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("-spe", "--steps",dest = "spe", help="Steps per Epoch")
parser.add_argument("-vspe", "--val_steps",dest = "vspe", help="Validation steps per epoch")
parser.add_argument("-epochs", "--epochs",dest = "epochs", help="No. of epochs")
parser.add_argument("-d", "--dataset",dest = "dataset", help="which dataset: asl or cdp")
parser.add_argument("-typ", "--model_type",dest = "model_type", help="Model type: child_, young_ or adult_")

args = parser.parse_args()
try:
    epochs = int(args.epochs)
except:
    epochs = 100
#print(batch_size)
try:
    spe = int(args.spe)
except:
    spe = 25
#print(mode)
try:
    vspe = int(args.vspe)
except:
    vspe = 10
#print(epochs)
try:
    dataset = args.dataset
except:
    dataset = 'cdp'
#print(device)
try:
    model_type = args.model_type
except:
    model_type = 'child_'


#%%
# Preparing data for cats_dogs_pandas dataset
if dataset=='asl':
    base_dir = '../Data/asl/'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
else:
    base_dir = '../Data/cats_dogs_pandas/'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
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
                                    vertical_flip=True,
                                    fill_mode='nearest'
                                    )
train_generator = train_datagen.flow_from_directory(
                                                    train_dir,
                                                    target_size=(256, 256),
                                                    batch_size=50,
                                                    class_mode='sparse'
                                                    )
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
                                                    val_dir,
                                                    target_size=(256, 256),
                                                    batch_size=25,
                                                    class_mode='sparse'
                                                    )
#%%
# Model

m = Models(dataset)

if model_type=='child_':
    model = m.child_()
elif model_type=='young_':
    model = m.young_()
else:
    model = m.adult_()

model.summary()

#%%
# Training the model
s = time.time()
history = model.fit_generator(
                                train_generator,
                                steps_per_epoch=spe, epochs=epochs, 
                                validation_data=val_generator, validation_steps=vspe,
                                verbose=1
                                )
e = time.time()
print('========================================================================')
time_taken = e-s
if time_taken>=60:
    print(f'Time taken to train for {epochs} epochs: {time_taken/60} minutes.')
elif time_taken>=3600:
    print(f'Time taken to train for {epochs} epochs: {time_taken/3600} hours.')
else:
    print(f'Time taken to train for {epochs} epochs: {time_taken} seconds.')
print('========================================================================')

#%%
# Predictions
pred_dir = '../Data/cats_dogs_pandas/test/'
imgs = glob.glob(pred_dir+'*.jpg')
images = []

for i in imgs:
    img = image.load_img(i, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images.append(np.vstack([x]))

preds = []
for img in images:
    preds.append(model.predict(img))
#print(preds)

# %%
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
plt.show()
