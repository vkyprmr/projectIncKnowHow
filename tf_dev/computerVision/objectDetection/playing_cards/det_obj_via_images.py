"""
Developer: vkyprmr
Filename: det_obj_via_images.py
Created on: 2020-11-18, Mi., 14:19:8
"""
"""
Modified by: vkyprmr
Last modified on: 2020-11-18, Mi., 15:2:31
"""

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import random
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


class DetectObjectsInImages:
    def __init__(self, model_path, label_path, img_path):
        device = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(device[0], True)
        except Exception as e:
            print(f'Error: {e}')

        warnings.filterwarnings('ignore')

        self.PATH_TO_MODEL = model_path
        self.PATH_TO_LABELS = label_path
        self.PATH_TO_IMAGES = img_path

    def load_model(self):
        """
        Loads the model saved after training.
        Returns:
            the detection function and category index needed for object detection

        """
        start = time.time()
        print(f'Loading model...')
        det_fnc = tf.saved_model.load(self.PATH_TO_MODEL)
        print(f'Done! Took {time.time() - start} secinds')
        category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)
        return det_fnc, category_index

    def load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))

    def detect_objects(self):
        """
        Detects and plots objects in given image/images.
        """
        det_fnc, category_index = self.load_model()
        pred_imgs = []
        for image_path in self.PATH_TO_IMAGES:
            print(f'Running inference for {image_path}... ')

            image_np = self.load_image_into_numpy_array(image_path)

            # Things to try:
            # Flip horizontally
            # image_np = np.fliplr(image_np).copy()

            # Convert image to grayscale
            # image_np = np.tile(
            #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            # input_tensor = np.expand_dims(image_np, 0)
            detections = det_fnc(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.50,
                agnostic_mode=False,
                line_thickness=10)

            # print(image_np_with_detections.shape)
            pred_imgs.append(image_np_with_detections)
            print('Done')

        if len(pred_imgs)>=4:
            plot_images = random.sample(pred_imgs, 4)
            i = 0
            for img in plot_images:
                sp = plt.subplot(2, 2, i + 1)
                # sp.axis('Off')  # Don't show axes (or gridlines)

                plt.imshow(img)
                plt.tight_layout()
                i += 1
        elif len(pred_imgs)==1:
            plt.imshow(pred_imgs[0])
            plt.tight_layout()
        else:
            i = 0
            for img in pred_imgs:
                sp = plt.subplot(1, 2, i + 1)
                # sp.axis('Off')  # Don't show axes (or gridlines)

                plt.imshow(img)
                plt.tight_layout()
                i += 1

        plt.show()
