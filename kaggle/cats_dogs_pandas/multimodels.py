'''
Developer: vkyprmr
Filename: multimodels.py
Created on: 2020-09-21 at 15:47:59
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-21 at 15:59:55
'''

# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# Class for building multiple models


class MultiModels:
    def __init__(self):
        print('Please provide a list of dictionaries containing your parameters.')
        """ 
        Arguments for build_models:
            model_params0 = {
                            'ip_filters': 16,
                            'kernel_size': (3,3),
                            'ip_pool': (2,2),
                            'range_nlayers': [32,64,128],
                            'n_dense': [256],
                            'op_shape': 3
                            }
            model_params1 = {
                            'ip_filters': 16,
                            'kernel_size': (3,3),
                            'ip_pool': (2,2),
                            'range_nlayers': [32,64,128,256],
                            'n_dense': [512,256],
                            'op_shape': 3
                            }
            model_params = [model_params0, model_params1]
         """

    def build_models(self, model_params):
        """ 
        Arguments:
            model_params0 = {
                            'ip_filters': 16,
                            'kernel_size': (3,3),
                            'ip_pool': (2,2),
                            'range_nlayers': [32,64,128],
                            'n_dense': [256],
                            'op_shape': 3
                            }
            model_params1 = {
                            'ip_filters': 16,
                            'kernel_size': (3,3),
                            'ip_pool': (2,2),
                            'range_nlayers': [32,64,128,256],
                            'n_dense': [512,256],
                            'op_shape': 3
                            }
            model_params = [model_params0, model_params1]
        Returns:
            A list containing all models.
         """
        built_models = []
        for m in model_params:
            ip_filters = m['ip_filters']
            kernel_size = m['kernel_size']
            ip_pool = m['ip_pool']
            range_nlayers = m['range_nlayers']
            n_dense=m['n_dense']
            op_shape = m['op_shape']

            mstr = ''
            for i in range_nlayers:
                mstr+=str(i)

            dstr = ''
            for d in n_dense:
                dstr+=str(d)
                
            layers = [
                    Conv2D(ip_filters, kernel_size, activation='relu', input_shape=(128,128,3)),
                    MaxPooling2D(pool_size=ip_pool)
                    ]

                

            #for m in model:	
            for i in range_nlayers:
                c_layer = Conv2D(i, kernel_size, activation='relu')
                mxpool = MaxPooling2D(pool_size=ip_pool)
                layers.append(c_layer)
                layers.append(mxpool)
                layers.append(Dropout(0.2))
        
            layers.append(Flatten())
            for dl in n_dense:
                layers.append(Dense(dl, activation='relu'))
                layers.append(Dropout(0.1))

            layers.append(Dense(op_shape, activation='softmax'))

            

            poolstr = f'{ip_pool[0]}x{ip_pool[1]}'

            model_name = f'{len(range_nlayers)}-{ip_filters}{kernel_size[0]}x{kernel_size[1]}_{mstr}-{dstr}_{poolstr}'

            model = Sequential(layers=layers, name=model_name)
            model.summary()
            print('\n')
            print('==================================================================')
            print('\n')
            built_models.append(model)

        print(f'Total number of models: {len(built_models)}')

        return built_models