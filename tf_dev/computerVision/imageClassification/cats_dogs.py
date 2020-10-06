"""
Developer: vkyprmr
Filename: cats_dogs.py
Created on: 2020-10-6, Di., 18:58:28
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-6, Di., 19:37:29
"""


# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Data
base_dir = '../../../Data/cats_vs_dogs/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

def sample_images(directory):
    nrows, ncols = 5, 5
    dogs = os.path.join(directory, 'dogs')
    cats = os.path.join(directory, 'cats')
    dog_files = os.listdir(dogs)
    cat_files = os.listdir(cats)
    pic_index = np.random.randint(10, len(cat_files))
    fig = plt.gcf()
    next_cat = [os.path.join(cats, cat_file) for cat_file in cat_files[pic_index-10:pic_index]]
    next_dog = [os.path.join(dogs, dog_file) for dog_file in dog_files[pic_index-10:pic_index]]
    for i, img_path in enumerate(next_cat + next_dog):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.show()









