import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

pd.options.display.max_columns = None


loc_df = pd.read_csv('stations.csv')
loc_df.head()

# data: https://www.kaggle.com/smid80/weatherww2
df = pd.read_csv('operations.csv') # STA: Weather Station

df.shape
df.head()

####################################################################################################
######################################## data cleaning #############################################
####################################################################################################

# null values in cols
cols_df = pd.DataFrame({
    'null_percentage': pd.isna(df).sum() / len(df) * 100,
    'col_type': df.dtypes
}).sort_values('null_percentage', ascending=False)
cols_df.head()

# remove empty cols
DROPPABLE_COLS = cols_df[cols_df['null_percentage'] > 90].index.tolist()
DROPPABLE_COLS

df = df.drop(columns=DROPPABLE_COLS)

# categorical fetures to numbers
cols_df[cols_df['col_type'] == 'object']

# date -> datetime
df['Date'] = pd.to_datetime(df['Date'])

df['PoorWeather'].value_counts(dropna=False) # repeat of TSHDSBRSGF
df = df.drop(['PoorWeather'], axis=1)

# TSHDSBRSGF expand
colnames = ['Thunder', 'Sleet', 'Hail', 'Dust', 'Smoke', 'Blowing_Snow', 'Rain', 'Snow', 'Glaze', 'Fog']

df['TSHDSBRSGF'].value_counts(dropna=False)

df['TSHDSBRSGF'] = df['TSHDSBRSGF'].replace({
    1: '1',
    np.nan: '0'
}).str.replace(' ', '0').apply(lambda x: '{:0<10}'.format(x))

events_splitted_df = df['TSHDSBRSGF'].apply(lambda x: pd.Series(
    dict(zip(colnames, list(str(x))))
)).astype(np.uint8)

df = pd.concat([df, events_splitted_df], axis=1)

# duplicates
DROPPABLE_COLS += [
    'YR', # year
    'MO', # month
    'DA', # day
    'PRCP', # Precip Inches
    'SNF', # Snowfall Inches
    'MAX', # MaxTemp Fahrenheit
    'MIN', # MinTemp Fahrenheit
    'MEA', # Mean temp Fahrenheit
    'Glaze', # always = 0
    'MeanTemp'
]

df = df.drop(columns=DROPPABLE_COLS, errors='ignore')


# rows cleaning
df['Precip'].value_counts(dropna=False) # Precipitation in mm
df['Snowfall'].value_counts(dropna=False) # Snowfall in mm

# Trace (T): very small value (see https://en.wikipedia.org/wiki/Trace_(precipitation) )
T = 0.01
df['Precip'] = df['Precip'].replace('T', T).astype('float').fillna(0)
df['Snowfall'] = df['Snowfall'].replace({'#VALUE!': -1}).fillna(0).astype('float')

# MinTemp > MaxTemp
df[df['MinTemp'] > df['MaxTemp']]
df = df.drop(index=df[df['MinTemp'] > df['MaxTemp']].index)
df.shape
df.columns

####################################################################################################
######################################## stats & plots #############################################
####################################################################################################

df.describe()
df.hist('Precip', figsize=(12, 8), color='red', bins=200)
df.boxplot('Precip')
plt.show()

sns.pairplot(df[[
    'Precip',
    'MaxTemp',
    'MinTemp',
    'Snowfall',
    'Thunder'
]])
plt.show()

# high correlation min max temp
df.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

# rain/snow depending on temp threshold
g = sns.pairplot(df[[
    'Precip',
    'MaxTemp',
    'MinTemp',
    'Snowfall',
]])
for ax in g.axes.ravel():
    ax.axvline(x=0, ls='--', linewidth=3, c='red')
plt.show()


####################################################################################################
########################################## simple lm ###############################################
####################################################################################################

X = df[['MinTemp']]
y = df['MaxTemp']

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# model training
sm = LinearRegression()
sm.fit(X_train, y_train)

sm.intercept_
sm.coef_

print(f'y = {sm.coef_} x + {sm.intercept_}') # one unit of change in Min => 0.92 change in Max

# model evaluation
y_pred = sm.predict(X_test)

res_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'MinTemp': X_test['MinTemp']
})

res_df

# visualize actual-predicted for a small sample
res_df30 = res_df.head(30)
res_df30.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# visualize relationship
ax = res_df.plot(x='MinTemp', y='Actual', kind='scatter', color='gray')
ax.set_xlabel('MinTemp')
ax.set_ylabel('MaxTemp')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

# metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


####################################################################################################
########################################### multi lm ###############################################
####################################################################################################
df.shape

TRAIN_COLS = [
    'Precip',
    'MaxTemp',
    'MinTemp'
]

TARGET_COL = 'Snowfall'

X = df[TRAIN_COLS]
y = df[TARGET_COL]

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

X_train[['Precip']].plot(kind='hist')
plt.show()

sns.displot(X_train[['MinTemp', 'MaxTemp']])
plt.show()


def get_model():

    imputer = SimpleImputer(strategy='median') # for each column apply median to null

    std_scaler_columns = [TRAIN_COLS.index(x) for x in ['MinTemp', 'MaxTemp']]
    minmax_scaler_columns = [TRAIN_COLS.index(x) for x in ['Precip']]
    ct = ColumnTransformer([
        ('stdscaler', StandardScaler(), std_scaler_columns), # apply Standard Scaler to Gaussian vars
        ('minmaxscaler', MinMaxScaler(), minmax_scaler_columns) # apply MinMax Scaler to skewed vars
    ], remainder='passthrough')

    # estimator
    estimator = LinearRegression()

    # pipeline
    pipe = Pipeline(steps=[
        ('imputer', imputer),
        ('col_transformer', ct),
        ('estimator', estimator)
    ])

    return pipe

pipe = get_model()
set_config(display='diagram')
pipe  # DAG style steps representation

pipe.fit(X_train, y_train)
pipe['estimator'].intercept_
pipe['estimator'].coef_


y_pred = pipe.predict(X_test)

print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred):.3f}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred):.3f}')
print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.3f}')

# overfitting assessment on training set
y_train_pred = pipe.predict(X_train)

print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_train, y_train_pred):.3f}') # ok: close to test
print(f'Mean Squared Error: {metrics.mean_squared_error(y_train, y_train_pred):.3f}') # ok: close to test
print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)):.3f}') # ok: close to test

# cross validation (test model on different train-test split)
kf = KFold(n_splits=5, shuffle=True, random_state=123)
'''
iter_1: TEST train train train train
iter_2: train TEST train train train
iter_3: train train TEST train train
iter_4: train train train TEST train
iter_5: train train train train TEST
'''

cv_results = []
for cvi, (train_index, test_index) in enumerate(kf.split(X)):
    print('*' * 30, cvi, '*' * 30)
    print('TRAIN:', len(train_index), 'TEST:', len(test_index))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipe = get_model()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    cv_results.append({
        'MAE_test': metrics.mean_absolute_error(y_test, y_pred),
        'MAE_train': metrics.mean_absolute_error(y_train, y_train_pred),
        'MSE_test': metrics.mean_squared_error(y_test, y_pred),
        'MSE_train': metrics.mean_squared_error(y_train, y_train_pred),
        'RMSE_test': metrics.mean_squared_error(y_test, y_pred),
        'RMSE_train': metrics.mean_squared_error(y_train, y_train_pred),
    })

cv_results_df = pd.DataFrame(cv_results)
cv_results_df # stability and no overfitting across folds


####################################################################################################
############################################ tuning ################################################
####################################################################################################

def test_models():

    imputer = SimpleImputer(strategy='median') # for each column apply median to null

    std_scaler_columns = [TRAIN_COLS.index(x) for x in ['MinTemp', 'MaxTemp']]
    minmax_scaler_columns = [TRAIN_COLS.index(x) for x in ['Precip']]
    ct = ColumnTransformer([
        ('stdscaler', StandardScaler(), std_scaler_columns), # apply Standard Scaler to Gaussian vars
        ('minmaxscaler', MinMaxScaler(), minmax_scaler_columns) # apply MinMax Scaler to skewed vars
    ], remainder='passthrough')

    # pipeline
    pipe = Pipeline(steps=[
        ('imputer', imputer),
        ('col_transformer', ct),
        ('estimator', None) # set to null
    ])

    return pipe

# test different algorithms
param_grid = {
    'estimator': [
        LinearRegression(),
        Ridge(alpha=0.5, random_state=42),
        Lasso(alpha=0.5, random_state=42),
        RandomForestRegressor(max_depth=5, random_state=42)
    ]
}

search = GridSearchCV(
    test_models(),
    param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=123),
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=3
)

search.fit(X, y) # full dataset

search.best_estimator_

search.best_score_

search.cv_results_

plot_res_df = pd.DataFrame([
    search.cv_results_['split0_test_score'],
    search.cv_results_['split1_test_score'],
    search.cv_results_['split2_test_score'],
    search.cv_results_['split3_test_score'],
    search.cv_results_['split4_test_score']
], columns=[r['estimator'].__class__.__name__ for r in search.cv_results_['params']])


plot_res_df.plot(xlabel='Fold', ylabel='TestScore', xticks=range(plot_res_df.shape[1]+1))
plt.show() # RandomForestRegressor


####################################################################################################
############################################ export ################################################
####################################################################################################
import pickle

final_pipe = get_model()
final_pipe.fit(X, y) # full dataset

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(final_pipe, f)
