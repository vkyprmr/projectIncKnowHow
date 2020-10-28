"""
Developer: vkyprmr
Filename: resize_images.py
Created on: 2020-10-26, Mon, 16:59:49
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-27, Tue, 16:18:39
"""

# Imports
import cv2
import glob
from tqdm import tqdm

# Directories and files
train_org = 'playing_cards/archive/train/'
test_org = 'playing_cards/archive/test/'

resized_train = 'playing_cards/train/'
resized_test = 'playing_cards/test/'


# train_images = glob.glob(train_org+'*.jpeg') + glob.glob(train_org+'*.jpg') + glob.glob(train_org+'*.png')
# test_images = glob.glob(test_org+'*.jpeg') + glob.glob(test_org+'*.jpg') + glob.glob(test_org+'*.png')


# Function to read image and resize it
def resize_images(org_dir, new_dir, keep_ratio=True, scale=0.25, max_height=720, max_width=1280, show_stats=True):
    """
    Args:
        org_dir: (string) the directory where images are located
        new_dir: (string) the directory where to save the new images
        keep_ratio: (bool) whether or not to keep the aspect ratio
        scale: (float) default 0.25
        max_height: (int) new height of the image (default=720) assuming landscape orientation
        max_width: (int) new width of the image (default=1280) assuming landscape orientation
        show_stats: (bool) display orignial and new image sizes (default=True)

    Returns: saves resized images in the new directory with the same image name

    """
    images = glob.glob(org_dir + '*.jpeg') + glob.glob(org_dir + '*.jpg') + glob.glob(org_dir + '*.png')
    images.sort()
    max_height = max_height
    max_width = max_width

    # ToDO: create only for downscaling

    # images = ['playing_cards/archive/test\\2C.jpg']

    for image in tqdm(images, desc='Resizing images'):
        img = cv2.imread(image, -1)
        height = img.shape[0]
        width = img.shape[1]
        image_name = image.split('/')[-1].split('\\')
        if width < height:
            orientation = 'portrait'
        else:
            orientation = 'landscape'

        max_size = (max_width, max_height)
        if orientation == 'portrait':
            max_size = (max_size[1], max_size[0])
        else:
            max_size = max_size
        overall_size = height*width
        print(f'\n Image: {image_name[1]}')
        print(f'Original size of the image: {height}x{width}')

        if max_size[1] < height or max_size[0] < width:
            resize_status = True
            if keep_ratio:
                sc = 0.25
                new_height = int(height*sc)
                new_width = int(width*sc)
                new_size = (new_width, new_height)
            else:
                new_size = max_size
        else:
            resize_status = False
            new_size = (width, height)

        if resize_status:
            print('Resizing')
            print(new_size)
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            if show_stats:
                print(f'New size of the image: {new_size[1]}x{new_size[0]}')
        else:
            print('Not resizing')
            img_resized = img

        # print(height <= max_height and width <= max_width)
        # print(height, max_height, width, max_width)

        img_path = new_dir + image_name[1]
        cv2.imwrite(img_path, img_resized)
        # cv2.imshow('image', img)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # break
    print('Successfully resized images.')


resize_images(train_org, resized_train, keep_ratio=True, show_stats=True)
