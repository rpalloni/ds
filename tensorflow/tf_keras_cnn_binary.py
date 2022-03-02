import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# dropbox zip

image_size = (180, 180) # size to resize images read from disk
batch_size = 32         # size of the batches of data

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dogs_cats_images',
    labels='inferred', # class names inferred from subfolders
    label_mode='binary',
    validation_split=0.2, # fraction of data to reserve for validation
    subset='training',
    seed=123,
    image_size=image_size,
    batch_size=batch_size, # 800 / 32 = 25 tensors
    # color_mode='grayscale'
)

len(list(train_ds.as_numpy_iterator())) # 25 tensors

for i in train_ds:
    print(tf.data.Dataset.from_tensor_slices(i))

list(train_ds.as_numpy_iterator())[0]   # 32 images-labels per tensor

# first image features (pixels rgb)
image1data = list(train_ds.as_numpy_iterator())[0][0][0] # [tensor][features/labels][image]
len(image1data)     # 180 rows
len(image1data[0])  # 180 cols
image1data[0, 0]    # rgb data in a pixel (out of 32400)
tf.keras.preprocessing.image.array_to_img(image1data)

plt.imshow(image1data.astype('uint8'))
plt.show()

# first image label (dummy)
image1label = list(train_ds.as_numpy_iterator())[0][1][0] # an image label
image1label

# plot a sample of images and labels
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(int(labels[i]))
        plt.axis('off')
        plt.show()


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dogs_cats_images',
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    # color_mode='grayscale'
)

# no images overlap
set(train_ds.file_paths).intersection(set(val_ds.file_paths))

# increase the diversity of training set
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.25)
    ]
)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    print(labels[0])
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')
        plt.show()


train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

# fit model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # image augmentation block
    x = data_augmentation(inputs)

    # rescale rgb between 0 and 1
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D()(x)

    # x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.SpatialDropout2D(0.2)(x)
    # x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    # x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes

    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=(180, 180, 3), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
# print(model.summary())

epochs = 500

callbacks = [keras.callbacks.ModelCheckpoint('callbacks/save_at_{epoch}.h5', save_best_only=True)]

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# model evaluation
fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(left=0, right=1, bottom=0, hspace=0.05, wspace=0.5)
ax[0].set_title('accuracy')
ax[1].set_title('loss')
ax[0].plot(model.history.history['accuracy'])                 # accuracy on training
ax[0].plot(model.history.history['val_accuracy'], color='r')  # accuracy on test
ax[1].plot(model.history.history['loss'])
ax[1].plot(model.history.history['val_loss'], color='r')
plt.show()


# predict class for new image
img = keras.preprocessing.image.load_img('dogs_cats_images/test.jpg', target_size=image_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(f'This image is {(100 * (1 - score))}% cat and {(100 * score)}% dog.')
