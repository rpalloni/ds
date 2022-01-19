'''
keras is the official high-level api of TF2 and uses TF backend -> tf.keras
models apis: Sequential, Functional, Subclassing
'''

# predict whether a wine is red or white by looking at its chemical properties, such as volatile acidity or sulphates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense # standard neural network

dbcoltype = {
    'fixed acidity': float,
    'volatile acidity': float,
    'citric acid': float,
    'residual sugar': float,
    'chlorides': float,
    'free sulfur dioxide': float,
    'total sulfur dioxide': float,
    'density': float,
    'pH': float,
    'sulphates': float,
    'alcohol': float,
    'quality': int}

dataw = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';', dtype=dbcoltype)
datar = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';', dtype=dbcoltype)

dataw.head()
dataw.shape
dataw.isnull().sum()

datar.head()
datar.shape
datar.isnull().sum()


# EDA
dataw.describe()
datar.describe()

fig, ax = plt.subplots(1, 2)
ax[0].hist(datar.alcohol, 10, facecolor='red', ec="black", lw=0.5, alpha=0.5, label="Red wine")
ax[1].hist(dataw.alcohol, 10, facecolor='yellow', ec="black", lw=0.5, alpha=0.5, label="White wine")
fig.subplots_adjust(left=0, right=1, bottom=0, hspace=0.05, wspace=0.5)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
ax[0].legend(loc='best')
ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(datar['quality'], datar["sulphates"], color="red", edgecolors="black", lw=0.5)
ax[1].scatter(dataw['quality'], dataw['sulphates'], color="yellow", edgecolors="black", lw=0.5)
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0, 10])
ax[1].set_xlim([0, 10])
ax[0].set_ylim([0, 2.5])
ax[1].set_ylim([0, 2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(datar['volatile acidity'], datar['alcohol'], c=datar['quality'])
ax[1].scatter(dataw['volatile acidity'], dataw['alcohol'], c=dataw['quality'])
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0, 1.7])
ax[1].set_xlim([0, 1.7])
ax[0].set_ylim([5, 15.5])
ax[1].set_ylim([5, 15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol")
fig.subplots_adjust(top=0.85, wspace=0.7)
plt.show()


# Binary classification
datar['type'] = 1 # red
dataw['type'] = 0 # white

wines = datar.append(dataw, ignore_index=True) # merge datasets (classes imbalanced data)
wines.head()
wines.shape

corr = wines.corr()
corr.style.background_gradient(cmap='coolwarm')


# Split train test samples (red imbalance)
X = wines.iloc[:, 0:11] # predictors
y = np.ravel(wines.type) # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


# Standardize data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Create the model: multi-layer neural network (perceptron)
'''
Create a linear (sequential) stack of dense layers: input, hidden, output
Dense layers implement the following operation:
output = activation(dot product(input, units) plus bias)
units = kernels = weights matrix
'''
model = Sequential() # create the model passing a list of layer instances to the constructor

# Add an input layer
model.add(Dense(12,
                activation='relu', # activation functions: relu, tanh, sigmoid
                input_shape=(11,), # neurons (columns)
                kernel_initializer='ones', # initialize columns weights to one
                bias_initializer='zero', # initializer error to zero
                use_bias=True))
model.output_shape
model.summary()

# Add one hidden layer
model.add(Dense(8, activation='relu'))
model.output_shape

# Add an output layer
model.add(Dense(1, activation='sigmoid')) # output a probability array
model.output_shape

model.summary()
model.get_config()
model.get_weights()


# Apply the model to data
model.compile(loss='binary_crossentropy', # loss functions: mean_squared_error, binary_crossentropy (log loss), categorical_crossentropy
              optimizer='adam', # optimization algorithms: Stochastic Gradient Descent (SGD), ADAM and RMSprop
              metrics=['accuracy']) # monitor the accuracy during the training

model.fit(X_train, y_train,
          epochs=5, # number of iterations
          batch_size=1, # number of samples
          verbose=1) # progress logger

y_pred = model.predict(X_test)

y_test[0:10]
y_pred[0:10].round()


# Evaluate model and imbalancement
score = model.evaluate(X_test, y_test, verbose=1)

print('loss:', score[0], 'accuracy:', score[1]) # good fit

confusion_matrix(y_test, y_pred.round()) # breakdown of predictions
precision_score(y_test, y_pred.round()) # accuracy of classifier
recall_score(y_test, y_pred.round()) # coverage of classifier
f1_score(y_test, y_pred.round()) # weighted average of precision and recall
cohen_kappa_score(y_test, y_pred.round()) # classification accuracy normalized by the imbalance of the classes in the data
