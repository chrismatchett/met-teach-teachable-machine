import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = 'flower_photos'
batch_size = 5
img_height = 240
img_width = 320

# 1 - Load the class names we made while creating the model
class_names = np.load('class_names.npy')

# 2 - Load the model we trained
model = tf.keras.models.load_model('flower_model.h5')

test_path = 'test_image.jpg'

img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# 3 - Make a prediction on the classification of the image file named test_image.jpg
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
