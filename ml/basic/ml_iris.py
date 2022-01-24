import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # using Pipeline to abstract training steps
from sklearn.preprocessing import LabelBinarizer, LabelEncoder # transformers for target
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder # transformers for features
from sklearn.tree import DecisionTreeClassifier, plot_tree # estimator
from sklearn.svm import SVC # estimator
from sklearn.multiclass import OneVsRestClassifier # estimator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score # metrics

iris = datasets.load_iris() # classes: 0-setosa, 1-virginica, 2-versicolor

iris # dict object with data and metadata

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target_num'] = iris.target
df['target_class'] = df['target_num'].apply(lambda x: iris.target_names[x])
df['target_is_setosa'] = (df['target_class'] == 'setosa').astype(int)

df.head()
df.describe()
df['target_class'].value_counts()
df['target_class'].value_counts().plot(kind='barh', color=['green','purple','blue'])

sns.pairplot(df, hue='target_class')
plt.show()

# train test
X = df[iris.feature_names]
y = df['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=123)
y.value_counts() / len(y)

# dummify
y_dummy = df['target_is_setosa']

y_dummy.value_counts() / len(y_dummy) # unbalanced classes

X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X, y_dummy,
                                                                            test_size=0.25, # 25% data in test
                                                                            shuffle=True, # mix rows
                                                                            stratify=y_dummy, # keep df shares
                                                                            random_state=123) # seed
y_train_dummy.value_counts() / len(y_train_dummy)
y_test_dummy.value_counts() / len(y_test_dummy)


# encode multiclass string to number
le = LabelEncoder()
y_train_le = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_le = pd.Series(le.transform(y_test), index=y_test.index)

le.classes_
le.inverse_transform([0, 1, 2])

# normalize features to remove scale bias due to features magnitude
ss = StandardScaler()
ss.fit(X_train)

ss.mean_
ss.scale_

X_train_ss = pd.DataFrame(ss.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_ss = pd.DataFrame(ss.transform(X_test), columns=X_test.columns, index=X_test.index)

# Binary Decision Tree
dtc = DecisionTreeClassifier(max_depth=2, random_state=123)
dtc.fit(X_train_dummy, y_train_dummy)

plot_tree(dtc)
plt.show() # binary tree rule: if petal length <= 2.6 than setosa
X.columns[2]

y_pred_dummy = dtc.predict(X_test_dummy)
y_pred_dummy


# Multiclass
ovr = OneVsRestClassifier(estimator=SVC(random_state=123))
ovr.fit(X_train_ss, y_train_le)

y_pred = ovr.predict(X_test_ss)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import metrics

pd.options.display.max_columns = None


loc_df = pd.read_csv('stations.csv')
loc_df.head()

df = pd.read_csv('operations.csv') # STA: Weather Station

df.shape
df.head()

################### data cleaning #######################

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
# NOTE: T: very small value (see https://en.wikipedia.org/wiki/Trace_(precipitation) )
df['Precip'].value_counts(dropna=False) # Precipitation in mm
df['Snowfall'].value_counts(dropna=False) # Snowfall in mm

# Trace
T = 0.01
df['Precip'] = df['Precip'].replace('T', T).astype('float').fillna(0)
df['Snowfall'] = df['Snowfall'].replace({'#VALUE!': -1}).fillna(0).astype('float')

# MinTemp > MaxTemp
df[df['MinTemp'] > df['MaxTemp']]
df = df.drop(index=df[df['MinTemp'] > df['MaxTemp']].index)
df.shape
df.columns

################### stats and plots #######################

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


################### simple linear model #######################

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
