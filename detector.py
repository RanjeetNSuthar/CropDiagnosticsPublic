import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class Detector:
    def __init__(self, img_path, detector):
        self.img_path = img_path
        self.detector = detector

    def resize_image(self, new_width=256, new_height=256):
        # takes input image and convert it into tensor object
        image = tf.io.read_file(self.img_path)
        # Turn the jpeg tensor image into numerical Tensor with 3 colour channels (Red, Green, Blue)
        image = tf.image.decode_jpeg(image, channels=3)
        # convert the colour channel value from (0-255) to (0-1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize image as (size X size)
        image = tf.image.resize(image, size=[new_width, new_height])
        return image

    def run_detector(self):
        # detector = hub.load(
        #     "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1").signatures['default']

        img = self.resize_image(1280, 856)
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[
            tf.newaxis, ...]

        result = self.detector(converted_img)

        result = {key: value.numpy() for key, value in result.items()}

        for i in result['detection_class_labels']:
            if i == 263:
                return True

        return False
