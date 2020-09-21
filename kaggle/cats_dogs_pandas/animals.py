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
#%matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Runtime device and memory
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True)

#%%
# Preparing data
base_dir = '../Data/cats_dogs_pandas/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'test')

### Individual class directories
train_dir_cats = os.path.join(train_dir, 'cats')
train_dir_dogs = os.path.join(train_dir, 'dogs')
train_dir_pandas = os.path.join(train_dir, 'pandas')

### Files
train_fnames_cats = os.listdir(train_dir_cats)
train_fnames_dogs = os.listdir(train_dir_dogs)
train_fnames_pandas = os.listdir(train_dir_pandas)

#%%

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
                                                    train_dir,
                                                    target_size=(64, 64),
                                                    batch_size=50,
                                                    class_mode='sparse'
                                                    )
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
                                                    val_dir,
                                                    target_size=(64, 64),
                                                    batch_size=25,
                                                    class_mode='sparse'
                                                    )

# %%
# Building the model
model_name = f'2C1D_16x32xd01x64x256_rms'

model = Sequential(layers=[
                            Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
                            MaxPooling2D(2,2),
                            Conv2D(32, (3,3), activation='relu'),
                            MaxPooling2D(2,2),
                            Dropout(0.1),
                            Conv2D(64, (3,3), activation='relu'),
                            MaxPooling2D(2,2),
                            Flatten(),
                            Dense(256, activation='relu'),
                            Dense(3, activation='softmax')
                          ],
                    name=model_name
                    )
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
model.summary()


#%%
# Training
### Callback functions

""" # Training
    epochs = 100
    lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
    history_lrs = model.fit(dataset, epochs=epochs, callbacks=[lr_schedule, tensorboard_callback], verbose=1)

    # Visualizing Learning Rates
    plt.semilogx(history_lrs.history['lr'], history_lrs.history["loss"])
    plt.axis([1e-8, 1e-3, 0, 300]) """

log_dir = 'logs/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')+'__'+model_name

ckpt_dir = 'logs/checkpoints/'+model_name+'/'
os.mkdir(ckpt_dir)
path_checkpoint = ckpt_dir+datetime.now().strftime('%Y%m%d-%H%M%S')+'.ckpt'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=25, verbose=1)

log_dir = 'logs/fit/'+datetime.now().strftime('%Y%m%d-%H%M%S')

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

models_dir = 'logs/models/'+model_name+'/'
os.mkdir(models_dir)
save_name = models_dir+datetime.now().strftime('%Y%m%d-%H%M%S')+'animals.h5'
model.save(save_name)

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

# %%

img = image.load_img('../Data/cats_dogs_pandas/images/panda.jpg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
preds = model.predict(images)
print(preds)

# %%
