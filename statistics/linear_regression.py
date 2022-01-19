import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as p
from sklearn.linear_model import LinearRegression

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
p.show()

###################################################################
##########################   ols code    ##########################
###################################################################
ozone = data[['Ozone']].values
wind = data[['Wind']].values
temp = data[['Temp']].values
wind
wind.shape
int = np.ones([data.shape[0], 1]) # create a array containing only ones
int
int.shape
X = np.concatenate([int, wind, temp], axis=1)
# X = np.hstack(int, solar)

np.dot(X.T,X) # https://en.wikipedia.org/wiki/Dot_product
# np.dot(X.T,X)[0,0]

inverse = np.linalg.inv(np.dot(X.T,X)) # (X'X)^-1
inverse
np.dot(X.T,ozone) # X'y

b = np.dot(inverse, np.dot(X.T,ozone)) # b = (X'X)^-1 * X'y
b # regression coef

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

# multi
predictors = data[['Wind', 'Temp']].values
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
predictors = data[['Wind', 'Temp']].values
model4 = LinearRegression()
model4.fit(predictors, ozone)
coeff_df = pd.DataFrame(model4.coef_[0], data[['Wind', 'Temp']].columns, columns=['Coefficient'])
coeff_df


###################################################################
##########################   gradient    ##########################
###################################################################

X = data[['Wind', 'Temp']].to_numpy()
y = data['Ozone'].to_numpy()

ss = StandardScaler()
X = ss.fit_transform(X)

lr = LinearRegression()
lr.fit(X, y)
lr.intercept_
lr.coef_



def predicted_y(weights, X, intercept):
    y_lst = []
    for i in range(len(X)):
        y_lst.append(weights@X[i]+intercept)
    return np.array(y_lst)

# linear loss
def loss(y, y_predicted):
    n = len(y)
    s = 0
    for i in range(n):
        s += (y[i]-y_predicted[i])**2
    return (1/n)*s

# derivative of loss on features
def dldw(x, y, y_predicted):
    n = len(y)
    s = 0
    for i in range(n):
        s += -(y[i]-y_predicted[i])*x[i]
    return (2/n)*s

# derivative of loss on intercept (bias)
def dldb(y, y_predicted):
    n = len(y)
    s = 0
    for i in range(len(y)):
        s += -(y[i]-y_predicted[i])
    return (2/n)*s

# gradient function
def gradient_descent(X, y, learn_rate=0.001, n_iter=3000):
    weights_vector = np.zeros(X.shape[1])
    intercept = 0

    for i in range(n_iter):
        y_predicted = predicted_y(weights_vector, X, intercept)
        weights_vector = weights_vector - learn_rate * dldw(X, y, y_predicted)
        intercept = intercept - learn_rate * dldb(y, y_predicted)

    return weights_vector, intercept

w, b = gradient_descent(X, y)

print("weights:", w)
print("intercept:", b)
