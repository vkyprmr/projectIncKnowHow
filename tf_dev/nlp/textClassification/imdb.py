"""
Developer: vkyprmr
Filename: imdb.py
Created on: 2020-10-10, Sa., 18:3:1
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-10, Sat, 20:38:26
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
data, meta_data = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train, test = data['train'], data['test']

x_train, y_train = [], []
x_test, y_test = [], []

for x, y in train:
    x_train.append(str(x.numpy()))
    y_train.append(y.numpy())

for x, y in test:
    x_test.append(str(x.numpy()))
    y_test.append(y.numpy())

y_train, y_test = np.array(y_train), np.array(y_test)

# Embedding
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_token = '<oov>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
s_train = tokenizer.texts_to_sequences(x_train)
s_train_padded = pad_sequences(s_train, maxlen=max_length, truncating=trunc_type)

s_test = tokenizer.texts_to_sequences(x_test)
s_test_padded = pad_sequences(s_test, maxlen=max_length)

# Decoder for reviews
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(s_train_padded[3]))
print(s_train[3])

# Building the model
model_name = f'imdb_embeddings-{vocab_size}_{embedding_dim}_{max_length}'
layers = [
    Embedding(vocab_size, embedding_dim, input_shape=(max_length,)),
    Flatten(),  # GlobalAveragePooling1D
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
]
model = Sequential(layers=layers, name=model_name)
opt = Adam(lr=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Preperation for training
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_name
chkpt_dir = 'logs/checkpoints_' + model_name + '/'
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

epochs = 100
history = model.fit(s_train_padded, y_train, epochs=epochs, validation_data=(s_test_padded, y_test), verbose=1)


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

# Storing data and visualize embeddings
# e = model.layers[0]
# weights = e.get_weights()[0]  # weights.shape = (vocab_size, embedding_dim)
#
# vectors_file = f'logs/embedding_data/vectors_{model_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.tsv'
# meta_data_file = f'logs/embedding_data/meta_data_{model_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.tsv'
# vectors_out = io.open(vectors_file, 'w', encoding='utf-8')
# meta_out = io.open(meta_data_file, 'w', encoding='utf-8')
#
# for i in range(1, vocab_size):
#     word = reverse_word_index[i]
#     embeddings = weights[i]
#     meta_out.write(f'{word}\n')
#     vectors_out.write('\t'.join([str(x) for x in embeddings]) + '\n')
# vectors_out.close()
# meta_out.close()
