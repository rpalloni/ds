# weather forecast app based on trained model
import pickle
import pandas as pd

# load model
with open('trained_model.pkl', 'rb') as f:
    trained_estimator = pickle.load(f)


# app core
def predict(precip, max_temp, min_temp):
    df = pd.DataFrame([{
        'Precip': precip,
        'MaxTemp': max_temp,
        'MinTemp': min_temp
    }])

    return trained_estimator.predict(df)


# input data
print(f'Snowfall predicted with received data: {predict(2, 26, 22)} cm')
print(f'Snowfall predicted with received data: {predict(10, 8, 2)} cm')
print(f'Snowfall predicted with received data: {predict(14, 2, -2)} cm') # the lower the temperature, the higher the snowfall cm
