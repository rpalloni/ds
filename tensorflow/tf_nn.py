# MNIST fashion images dataset: https://www.tensorflow.org/datasets/catalog/fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images.shape # 60k grayscale (1 color dimension) images with 28x28 pixels

train_images[0]

train_images[0, 23, 23] # pixel value: 0 (black) - 255 (white)

train_labels.shape

train_labels[:10] # clothing articles: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot

class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

plt.figure()
plt.imshow(train_images[1], cmap='Greys')
plt.colorbar()
plt.show()
