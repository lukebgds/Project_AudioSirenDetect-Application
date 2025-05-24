import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def preprocess_image(img_path):

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_final = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_final
