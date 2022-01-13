import io
import requests
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as p


download_url = "https://raw.githubusercontent.com/rpalloni/dataset/master/airquality.csv"
response = requests.get(download_url)
# data = pd.read_csv(io.BytesIO(response.content), dtype={'Ozone': float,'SolarRay':float, 'Wind':float, 'Temp':float, 'Month':float, 'Day':float})
data = pd.read_csv("https://raw.githubusercontent.com/rpalloni/dataset/master/airquality.csv", 
            dtype={'Ozone': float,'SolarRay':float, 'Wind':float, 'Temp':float, 'Month':float, 'Day':float})
data.head()
data.shape

print(data['Ozone']) # series

data.isnull().values.any()
data = data.dropna()
# data = pd.DataFrame(data)

data.describe()

data.hist('Ozone', figsize=(12,8), color='red', bins = 20)
data.hist(figsize=(10,10), color='red') # all vars

###################################################################
##########################   ols code    ##########################
###################################################################
ozone = data[['Ozone']].values
solar = data[['SolarRay']].values
solar
solar.shape
int = np.ones([data.shape[0], 1]) # create a array containing only ones
int
int.shape
X = np.concatenate([int,solar], axis=1)
# X = np.hstack(int, solar)

np.dot(X.T,X) # https://en.wikipedia.org/wiki/Dot_product

np.dot(X.T,X)[0,0]
np.dot(X.T,X)[0,1]
np.dot(X.T,X)[1,0]
np.dot(X.T,X)[1,1]

inverse = np.linalg.inv(np.dot(X.T,X)) # (X'X)^-1
inverse
np.dot(X.T,ozone) # X'y

b = np.dot(inverse, np.dot(X.T,ozone)) # b = (X'X)^-1 * X'y
b # regression coef


###################################################################
########################   scipy.stats    #########################
###################################################################
result = scipy.stats.linregress(data['SolarRay'], data['Ozone'])

result.slope
result.intercept
result.rvalue
result.pvalue # https://www.statsdirect.com/help/basics/p_values.htm
result.stderr

# scatterplot
line = f'Regression line: y={result.intercept:.2f}+{result.slope:.2f}x, r={result.rvalue:.2f}'

fig, ax = p.subplots()
ax.plot(data['SolarRay'], data['Ozone'], linewidth=0, marker='s', label='Data points')
ax.plot(data['SolarRay'], result.intercept + result.slope * data['SolarRay'], label=line)
ax.set_xlabel('solar')
ax.set_ylabel('ozone')
ax.legend(facecolor='white')
p.show()

###################################################################
########################   statsmodels    #########################
###################################################################
predictors = data[['SolarRay']].values
predictors = sm.add_constant(predictors) # add intercept
predictors

model0 = sm.OLS(ozone, predictors) # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
res = model0.fit() # results wrapper with many result data
print(res.summary()) # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
res.rsquared
res.params
res.fittedvalues

# x1 x2 x3
predictors = data[['SolarRay', 'Wind', 'Temp']].values
predictors = sm.add_constant(predictors) # add intercept
predictors

model1 = sm.OLS(ozone, predictors) # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
res = model1.fit() # results wrapper with many result data
print(res.summary()) # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
res.rsquared
res.params
res.fittedvalues

###################################################################
##########################   sklearn    ###########################
###################################################################
model3 = LinearRegression() # instantiate model
ozone = data[['Ozone']].values
#ozone = data['Ozone'].values.reshape(-1,1)
solar = data[['SolarRay']].values
#solar = data['SolarRay'].values.reshape(-1,1)

print(ozone) # vector

p.scatter(ozone, solar)

model3.fit(solar, ozone)
model3.score(solar, ozone) # r2
model3.intercept_
model3.coef_
model3.predict(ozone)
model3.intercept_ + model3.coef_ * ozone

coeff_df = pd.DataFrame(model3.coef_, data[['SolarRay']].columns, columns=['Coefficient'])
coeff_df

# multi
predictors = data[['SolarRay', 'Wind', 'Temp']].values
model4 = LinearRegression()
model4.fit(ozone, predictors)
coeff_df = pd.DataFrame(model4.coef_, data[['SolarRay', 'Wind', 'Temp']].columns, columns=['Coefficient'])
coeff_df
