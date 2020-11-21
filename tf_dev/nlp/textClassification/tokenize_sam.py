"""
Developer: vkyprmr
Filename: tokenize.py
Created on: 2020-10-9, Fr., 14:34:28
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-9, Fr., 14:44:53
"""

# Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Enabling dynamic GPU usage
device = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(device[0], True)
except Exception as e:
    print(f'Error: {e}')

sentences = [
    'I love to travel',
    'I love to cook',
    'I am a machine learning enthusiast',
    'I work as a Machine Learning Engineer'
]

tokenizer = Tokenizer(oov_token='<oov>')    # oov: out of vocabulary tokens
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(f'Word index: {word_index}')
seq = tokenizer.texts_to_sequences(sentences)
print(f'Sequences:\n{seq}')

padded_seq = pad_sequences(seq, padding='post')
# max_length=5, truncating='post' -- default padding='pre', truncating='pre'
print(padded_seq)
