"""
Developer: vkyprmr
Filename: imdb_8k.py
Created on: 2020-10-10, Sa., 21:24:29
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-13, Di., 17:11:24
"""

# Imports
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Enabling dynamic GPU usage
device = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(device[0], True)
except Exception as e:
    print(f'Error: {e}')

# Data
data, meta_data = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train, test = data['train'], data['test']

# x_train, y_train = [], []
# x_test, y_test = [], []
#
# for x, y in train:
#     x_train.append(x.numpy())
#     y_train.append(y.numpy())
#
# for x, y in test:
#     x_test.append(x.numpy())
#     y_test.append(y.numpy())
#
# y_train, y_test = np.array(y_train), np.array(y_test)

# Embedding
tokenizer = meta_data.features['text'].encoder

vocab_size = tokenizer.vocab_size
embedding_dim = 64

# Building the model
model_name = f'imdb_8k-{vocab_size}_{embedding_dim}'
layers = [
    Embedding(vocab_size, embedding_dim),
    GlobalAveragePooling1D(),  # GlobalAveragePooling1D, Flatten
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
]
model = Sequential(layers=layers, name=model_name)
opt = Adam(lr=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Preperation for training
log_dir = "..\\logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_name
# chkpt_dir = '../logs/checkpoints_' + model_name + '/'
# if not os.path.exists(chkpt_dir):
#     os.mkdir(chkpt_dir)
#
# path_chkpt = chkpt_dir + datetime.now().strftime('%Y%m%d-%H%M%S')
# tb_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)
# es_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, verbose=1)
# chkpt_callback = ModelCheckpoint(filepath=path_chkpt, monitor='val_loss',
#                                  verbose=1, save_weights_only=True,
#                                  save_best_only=True)
# callbacks = [tb_callback, es_callback, rlr_callback, chkpt_callback]

epochs = 100
history = model.fit(train, epochs=epochs,
                    validation_data=test,
                    verbose=1)


# ToDo: fix shape error raising while training

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
