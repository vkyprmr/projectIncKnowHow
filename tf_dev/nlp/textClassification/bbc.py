"""
Developer: vkyprmr
Filename: bbc.py
Created on: 2020-10-10, Sa., 17:42:45
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-10, Sa., 17:55:3
"""

# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Enabling dynamic GPU usage
device = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(device[0], True)
except Exception as e:
    print(f'Error: {e}')

# Data
file_loc = '../../../Data/bbc/bbc_text.csv'
df = pd.read_csv(file_loc)

sentences = df.text.to_list()
labels = df.category.to_list()

# Tokenizing
text_tokenizer = Tokenizer(oov_token='<oov>')
text_tokenizer.fit_on_texts(sentences)
text_seq = text_tokenizer.texts_to_sequences(sentences)
text_padded = pad_sequences(text_seq)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_seq = label_tokenizer.texts_to_sequences(labels)
