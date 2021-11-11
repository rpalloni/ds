'''
keras is the official high-level api of TF2 and uses TF backend -> tf.keras
models apis: Sequential, Functional, Subclassing
'''

import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense # standard neural network
from keras.layers import LSTM # long-short term memory layer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('air_pollution.csv', sep=' ')
df.head()

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time']).dt.floor('h')
df.drop(['Date', 'Time'], axis=1, inplace=True)

df.groupby('DateTime').mean()
