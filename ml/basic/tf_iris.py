import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

data = tfds.load('iris', split='train')
df = tfds.as_dataframe(data) # pandas df

df

FEATURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
LABEL = ['Setosa', 'Versicolor', 'Virginica']

ds = pd.DataFrame(df['features'].to_list(), columns=FEATURES)
ds['Species'] = df['label']

split = round(len(ds) * 0.80)
dftrain, dftest = ds[:split], ds[split:]


y_train = dftrain.pop('Species') # extract and drop var from df
y_test = dftest.pop('Species')

dftrain.describe()
y_train.describe()
y_test.describe()


def input_fn(features, label, training=True, batch_size=256):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), label))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


feature_columns = []
for key in FEATURES:
    feature_columns.append(tf.feature_column.numeric_column(key=key))


# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each
# https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)


classifier.train(input_fn=lambda: input_fn(dftrain, y_train, training=True), steps=5000)

test_result = classifier.evaluate(input_fn=lambda: input_fn(dftest, y_test, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**test_result))


# Predict class of new data
predict = {}

print("Please type numeric values: ")
for feature in FEATURES:
    val = input(feature + ": ")
    predict[feature] = [float(val)]


predictions = classifier.predict(input_fn=lambda: tf.data.Dataset.from_tensor_slices(dict(predict)).batch(30))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

print('Prediction is "{}" ({:.1f}%)'.format(LABEL[class_id], 100 * probability))

# input test
# 5.1, 3.3, 1.7, 0.5 => setosa
# 6.9, 3.1, 5.4, 2.1 => virginica
