import PIL
import pathlib
import PIL.Image
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

'''
flowers dataset contains five sub-directories, one per class:
flower_photos
 |- daisy
 |- dandelion
 |- roses
 |- sunflowers
 |- tulips
'''

dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))


batch_size = 32
img_height = 180
img_width = 180


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset='training',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset='validation',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# no images overlap
set(train_ds.file_paths).intersection(set(val_ds.file_paths))

class_names = train_ds.class_names
class_names # 0/1/2/3/4


# ~92 tensors (2936/32)
len(list(train_ds.as_numpy_iterator()))
for images, labels in train_ds:
    print(images.shape, labels.shape)

# each tensor (batch) has shape (32, 180, 180, 3): 32 images of height 180,
# width 180 and 3 rgb data, and a tensor of shape (32,): 32 labels
list(train_ds.as_numpy_iterator())[0]

# 32 labels
images_first_tensor__labels = list(train_ds.as_numpy_iterator())[0][1]
pd.DataFrame(images_first_tensor__labels, columns=['labels'])


images_first_tensor__features = list(train_ds.as_numpy_iterator())[0][0]
len(images_first_tensor__features) # 32 images-labels per tensor
# 1 image - 32400 pixels - 97200 rgb data
pixels = [item for items in images_first_tensor__features[0] for item in items] # flatten nested list
rgbdata = [item for items in pixels for item in items]
pd.DataFrame(rgbdata)


# each tensor is a 32x97200 dataframe (i.e. 32x180x180x3)
# last tensor first image features (pixels rgb)
image1data = list(train_ds.as_numpy_iterator())[91][0][0] # [tensor][features][image]
len(image1data)     # 180 rows
len(image1data[0])  # 180 cols
image1data[0, 0]    # 3 rgb data in a pixel => 97200 data
image1data[179, 179]
tf.keras.preprocessing.image.array_to_img(image1data)

plt.imshow(image1data.astype('uint8'))
plt.show()

# first image label (dummy)
image1label = list(train_ds.as_numpy_iterator())[0][1][0] # an image label
image1label

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    plt.show()
