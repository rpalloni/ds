# MNIST fashion images dataset: https://www.tensorflow.org/datasets/catalog/fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images.shape # 60k grayscale images: each image is a 28x28 grid of pixels with a value
train_images[0] # each image is a row with 784 columns (28 arrays of 28 elements)
sum(len(train_images[i]) for i in train_images[0])
train_images[0, 23, 23] # pixel value: 0 (black) - 255 (white)


train_labels.shape # 60k clothing articles codes
train_labels[:10] # clothing articles: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot


plt.figure()
plt.imshow(train_images[0], cmap='Greys')
plt.colorbar()
plt.show()


class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
