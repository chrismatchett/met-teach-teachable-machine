import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1 - Prepare your dataset
# Our dataset of training images are in a folder called, 'flower_photos', with a resolution of 320x240px
data_dir = 'flower_photos'
batch_size = 32
img_height = 240
img_width = 320

# 2- Let's split our dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 3 - We will now define our model
num_classes = len(train_ds.class_names)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4 - We will now train our model
# The epochs parameter specifies the number of times the training data is passed forward and backward through the neural network
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 5 - We will now save the model as an .h5 binary file so we can use it later
model.save('flower_model.h5')

# 6 - We will now save our classification names as a binary file for faster loading
class_names = train_ds.class_names
np.save('class_names.npy', class_names)
