import numpy as np
import pandas as pd
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dftest = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

dftrain.shape
dftest.shape

y_train = dftrain.pop('survived') # extract and drop var from df
y_test = dftest.pop('survived')

dftrain.describe()

dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# features
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() # get a list of all unique values for each col
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def input_function(df, y, epochs=1000, batch_size=3):
  return tf.data.Dataset.from_tensor_slices((dict(df), y)).repeat(epochs).batch(batch_size)

# function to provide input data for training as minibatches
# x=lambda:print('hello')
# x
def train_input_fn(): return input_function(dftrain, y_train)
def test_input_fn(): return input_function(dftest, y_test, epochs=1, batch_size=4)


le = tf.estimator.LinearClassifier(feature_columns=feature_columns) # estimator
le.train(train_input_fn)
result = le.evaluate(test_input_fn)
print(result['accuracy'])

y_pred = le.predict(test_input_fn)
list(y_pred) # a dict for each prediction
