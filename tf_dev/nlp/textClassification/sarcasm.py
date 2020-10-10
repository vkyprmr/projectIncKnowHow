"""
Developer: vkyprmr
Filename: sarcasm.py
Created on: 2020-10-10, Sa., 16:14:53
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-10, Sa., 17:42:12
"""

# Imports
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

np.set_printoptions(linewidth=200)

# Enabling dynamic GPU usage
device = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(device[0], True)
except Exception as e:
    print(f'Error: {e}')

# Data
file_loc = '../../../Data/sarcasm/sarcasm_headlines_v2.json'
df = pd.read_json(file_loc, lines=True)

sentences = df.headline.to_list()
labels = df.is_sarcastic.to_list()
urls = df.article_link.to_list()

# Tokenizing
tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
seq = tokenizer.texts_to_sequences(sentences)
padded_seq = pad_sequences(seq, padding='post')
