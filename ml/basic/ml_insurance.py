import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/insurance.csv')
data.info()
data.shape

################### EDA #######################

sns.set_style('whitegrid', {'grid.linestyle': '--'})
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='premium', data=data, hue='sex')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.title('Distribution of premium by age and sex')
plt.show()

smokers = data['smoker'].unique()
colors = ['Reds', 'Greens']
for i, smoker in enumerate(smokers):
    temp = data[data['smoker'] == smoker]
    sns.scatterplot(temp['bmi'], temp['premium'], cmap=colors[i])
plt.legend(smokers)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='region', y='premium', hue='sex', data=data)
plt.show()


sns.set_style('whitegrid', {'grid.linestyle': '--'})
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='premium', data=data, hue='smoker')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Distribution of charges by age and sex')
plt.show()


plt.figure(figsize=(10, 8))
sns.boxplot(x='children', y='premium', hue='smoker', data=data)
plt.title('Distribution of charges by number of children')
plt.show()


#################### LR #######################

# assess multicollinearity
sns.heatmap(data.corr(), annot=True)
plt.show()


# transforming categorical features to numerical values
data['smoker'] = data['smoker'].replace(['yes', 'no'], [1, 0])
data['sex'] = data['sex'].replace(['male', 'female'], [1, 0])
data['region_southeast'] = data['region'].apply(lambda x: 1 if x == 'southeast' else 0)

# split train test
y_data = data['premium']
x_data = data.drop(['premium', 'region'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

# training
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)

# coefficients
model_coef = pd.DataFrame(data=model1.coef_, index=x_test.columns)
model_coef.loc['intercept', 0] = model1.intercept_
model_coef

# model performance
model_performance = pd.DataFrame(data=[r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))], index=['R2', 'RMSE'])
model_performance

residual = y_test - y_pred
# positive residual: actual premium > predicted premium
# negative residual: actual premium < predicted premium
plt.scatter(y_test, residual)
plt.title('Residual vs actual charges')
plt.xlabel('Actual charges')
plt.ylabel('Residual')
plt.show()
