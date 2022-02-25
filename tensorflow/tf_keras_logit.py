import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense # standard neural network
import tensorflow as tf
tf.random.set_seed(0) # set reproducible results

df = pd.read_csv('suv_data.csv')
df.head()
df.shape
df = df.drop(columns='User ID')
df['Gender'] = df['Gender'].replace(['Male', 'Female'], [1, 0])

####################### EDA ######################
df['Purchased'].value_counts() / len(df) # unbalanced classes
df.describe()

df.hist('Age', figsize=(8, 6), bins=20) # skewed hh income
plt.show()

df.hist('EstimatedSalary', figsize=(8, 6), bins=20) # skewed hh income
plt.show()


y, X = df['Purchased'].values, df.drop(columns='Purchased').values
scaler = StandardScaler()
X = scaler.fit_transform(X)


# model layers in sequential order
model = Sequential()
# first layer
model.add(Dense(units=12,               # output shape
                activation='linear',    # linear model
                input_shape=(3,)))      # input shape
# second layer
model.add(Dense(units=1,                # output shape
                activation='sigmoid'))  # logit model (softmax for multiclass)

model.summary()

# define model loss and optimizer
model.compile(loss='binary_crossentropy',   # loss functions: mean_squared_error (numeric), binary_crossentropy (dummy), categorical_crossentropy (multiclass)
              optimizer='adam',             # optimization algorithms: Stochastic Gradient Descent (SGD), ADAM and RMSprop
              metrics=['accuracy'])         # monitor the accuracy during the training


X, X_val, y, y_val = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=123)

e = 50
model.fit(x=X, y=y,
          epochs=e,
          batch_size=100,
          validation_data=(X_val, y_val),   # train test split
          verbose=0)                        # progress logger

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


y_pred = model.predict(X_val)
y_pred[0:10].round()
y_val[0:10]
