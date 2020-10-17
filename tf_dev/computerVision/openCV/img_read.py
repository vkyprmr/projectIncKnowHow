"""
Developer: vkyprmr
Filename: img_read.py
Created on: 2020-10-17, Sa., 14:37:2
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-17, Sa., 14:47:26
"""

# Imports
import numpy as np
from PIL import Image
np.set_printoptions(linewidth=200)

# Image
pic = 'C:/Users/vkypr/Pictures/ProPic.jpg'
im = Image.open(pic)
img = im.convert('1')
img = img.resize((28,28))
img = np.array(img).astype(int)
print(img)
