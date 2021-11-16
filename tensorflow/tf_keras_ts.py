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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('air_pollution.csv', sep=' ')
df.head()
df.shape

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']).dt.floor('h')
df.drop(['Date', 'Time'], axis=1, inplace=True)

df.set_index('DateTime')


# EDA
df.isnull().sum()
df.groupby('DateTime').mean()

df.describe()

# timeseries
plt.plot(df['DateTime'], df['No'])
plt.plot(df['DateTime'], df['O3'])
plt.plot(df['DateTime'], df['NO2'])
plt.plot(df['DateTime'], df['SO2'])
plt.plot(df['DateTime'], df['NO'])
plt.plot(df['DateTime'], df['CO2'])
plt.plot(df['DateTime'], df['VOC'])
plt.plot(df['DateTime'], df['PM1'])
plt.plot(df['DateTime'], df['PM2.5'])
plt.plot(df['DateTime'], df['PM4'])
plt.plot(df['DateTime'], df['PM10'])
plt.plot(df['DateTime'], df['TSP']) # total suspended particels

plt.plot(df['DateTime'], df['HUM'])
plt.plot(df['DateTime'], df['TEMP'])
plt.plot(df['DateTime'], df['WS'])
plt.plot(df['DateTime'], df['WD'])
plt.plot(df['DateTime'], df['ISPU'])

# histogram
fig, ax = plt.subplots()
ax.hist(df['TEMP'], bins=100, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# training and testing data
train_size = int(df.shape[0] * 0.75)
X_train = df[0:train_size]
X_test = df[train_size:df.shape[0]]

y_train = X_train['TEMP']
y_test = X_test['TEMP']

X_train = X_train.drop(['TEMP', 'DateTime'], axis=1)
X_test = X_test.drop(['TEMP', 'DateTime'], axis=1)


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ts reshape(samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

y_train = np.reshape(y_train, (y_train.shape[0],))
y_test = np.reshape(y_test, (y_test.shape[0],))

# LSTM Neural Network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 17)))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
m = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

# summarize history for accuracy
plt.plot(m.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(m.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

score = model.evaluate(X_test, y_test, verbose=1)

print('loss:', score[0], 'accuracy:', score[1])
