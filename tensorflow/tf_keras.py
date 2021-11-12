'''
keras is the official high-level api of TF2 and uses TF backend -> tf.keras
models apis: Sequential, Functional, Subclassing
'''

import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense # standard neural network
from keras.layers import LSTM # long-short term memory layer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('air_pollution.csv', sep=' ')
df.head()

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']).dt.floor('h')
df.drop(['Date', 'Time'], axis=1, inplace=True)

df.groupby('DateTime').mean()

df.describe()

target = df['TEMP']
predictors = df.drop('TEMP', axis=1)

# LSTM Neural Network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))

# split data into training and test datasets
trainX, testX, trainY, testY = train_test_split(predictors, target, test_size=0.25, random_state=123)

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=32, verbose=2)
