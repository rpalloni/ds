'''
keras is the official high-level api of TF2 and uses TF backend -> tf.keras
models apis: Sequential, Functional, Subclassing
'''

# predict temperature based on air chemical pollution properties over time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

df.set_index('DateTime')
df.groupby('DateTime').mean()

df.describe()

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


X = df.drop(['TEMP', 'DateTime'], axis=1)# predictors
y = np.ravel(df['TEMP']) # target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# data.reshape(samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# LSTM Neural Network
model = Sequential()
model.add(LSTM(4, input_shape=(None, 1, X.shape[1])))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1)
