import tensorflow as tf
import tensorflow_datasets as tfds

data = tfds.load('iris', split='train')
df = tfds.as_dataframe(data)
df

FEATURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
LABEL = ['Setosa', 'Versicolor', 'Virginica']

df['SepalLength'], df['SepalWidth'], df['PetalLength'], df['PetalWidth'] = zip(*df['features'])
df


dftrain = df[0:120]
dftest = df[120:150]

y_train = dftrain.pop('label') # extract and drop var from df
y_test = dftest.pop('label')
