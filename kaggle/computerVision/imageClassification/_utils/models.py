'''
Developer: vkyprmr
Filename: models.py
Created on: 2020-09-24 at 14:20:52
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-25 at 12:08:04
'''

#%%
# Imports
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

""" # Runtime device and memory:
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True) """

#%%


class Models:
    """
    Module to import predefined models.
     Arguments:
        which_dataset: either asl--> American sign language
                        or cdp --> Cats, dogs, pandas
     """
    def __init__(self, which_dataset='asl'):
        print('Select either child_, young_ or adult_. Inputs are considered of 128x128x3')
        if which_dataset=='asl':
            self.ops = 29
        else:
            self.ops = 3

    def child_(self):
        """ 
        Returns the base model.
        Layers:
            Input conv. layer with 16 filters and 3x3 kernel
            maxpooling layer with 2x2 pool
            conv. layer with 32 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.1
            flatten
            fully connected layer (dense) with 128 neurons and the
            output layer with 3 neurons for each class and softmax activation function.
            All layers except output, uses relu
            Loss: sparse_categorical_crossentropy
            Optimizer RMSProp(0.01)
         """
        self.child_name = 'child_model'
        
        self.child_model = Sequential(layers=[
                                                Conv2D(50, (3,3), activation='relu', input_shape=(256,256,3)),
                                                MaxPooling2D(2,2),
                                                Conv2D(64, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.1),
                                                Flatten(),
                                                Dense(128, activation='relu'),
                                                Dense(self.ops, activation='softmax')        
                                                ],
                                     name=self.child_name
                                                )
        self.child_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.01),
                                metrics=['accuracy'])
        return self.child_model

    def young_(self):
        """ 
        Returns a deeper model than the child model.
        Layers:
            Input conv. layer with 32 filters and 3x3 kernel
            maxpooling layer with 2x2 pool
            conv. layer with 64 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.2
            conv. layer with 128 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.25
            conv. layer with 64 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.15
            flatten
            fully connected layer (dense) with 32 neurons and the
            output layer with 3 neurons for each class and softmax activation function.
            All layers except output, uses relu
            Loss: sparse_categorical_crossentropy
            Optimizer RMSProp(0.01)
         """
        self.young_name = 'young_model'
        
        self.young_model = Sequential(layers=[
                                                Conv2D(25, (3,3), activation='relu', input_shape=(256,256,3)),
                                                MaxPooling2D(2,2),
                                                Conv2D(32, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.2),
                                                Conv2D(64, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.25),
                                                Conv2D(128, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.15),
                                                Flatten(),
                                                Dense(64, activation='relu'),
                                                Dense(self.ops, activation='softmax')        
                                                ],
                                     name=self.young_name
                                                )
        self.young_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.01),
                                metrics=['accuracy'])
        return self.young_model
        
    def adult_(self):
        """ 
        Returns a deeper model than the child model.
        Layers:
            Input conv. layer with 16 filters and 3x3 kernel
            maxpooling layer with 2x2 pool
            conv. layer with 32 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.1
            conv. layer with 64 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.15
            conv. layer with 128 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.2
            conv. layer with 64 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.15
            conv. layer with 32 filters and 3x3 kernel and pooled by 2x2 max pooling
            a dropout of 0.1
            flatten
            fully connected layer (dense) with 16 neurons and the
            output layer with 3 neurons for each class and softmax activation function.
            All layers except output, uses relu
            Loss: sparse_categorical_crossentropy
            Optimizer RMSProp(0.01)
         """
        self.adult_name = 'adult_model'
        
        self.adult_model = Sequential(layers=[
                                                Conv2D(50, (3,3), activation='relu', input_shape=(256,256,3)),
                                                MaxPooling2D(2,2),
                                                Conv2D(64, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.1),
                                                Conv2D(128, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.15),
                                                Conv2D(64, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.2),
                                                Conv2D(128, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.15),
                                                Conv2D(64, (3,3), activation='relu'),
                                                MaxPooling2D(2,2),
                                                #Dropout(0.1),
                                                Flatten(),
                                                Dense(32, activation='relu'),
                                                Dense(self.ops, activation='softmax')        
                                                ],
                                     name=self.adult_name
                                                )
        self.adult_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.01),
                                metrics=['accuracy'])
        return self.adult_model
