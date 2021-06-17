import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

###################################################################
##########################   nolib    ##########################
###################################################################

df = pd.DataFrame({'age': [22,25,47,52, 46,56,55,60,62,61,18,28,27,29,49,55,25,58,19,18,21,26,40,45,50,54,23,24,45,59],
                    'has_insurance':[0,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,0]})
df.shape

age = df[['age']]
intercept = np.ones((age.shape[0], 1))
ins = df[['has_insurance']]

features = np.hstack((intercept, age))
type(features)
type(ins)

weights = np.array([0.5,0.5]) # start guess

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# likelihood step by step
def log_likelihood(y, yp,n):
    ll = np.sum( y*np.log(yp) + (1-y)*np.log(1-yp) ) * (-1/n)
    return ll

scores = np.dot(features, weights)
pred =sigmoid(scores)
pred = pd.DataFrame({'pred':pred})

log_likelihood(ins.values, pred.values, len(ins)) # ll starting value

# likelihood full calc
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights) # nx2 * 2x1 = nx1
    pred = sigmoid(scores)
    pred = pd.DataFrame({'pred':pred}) # align type
    ll = np.sum( ins.values * np.log(pred.values) + (1-ins.values) * np.log(1 - pred.values) ) * (-1/len(ins))
    return ll

log_likelihood(features, ins, weights)

# Maximum Likelihood Estimation
# solve argmax with numerical optimizaion
def logistic_regression(predictors, target, num_steps, learning_rate):

    weights = np.array([0.5,0.5]) # np.zeros(features.shape[1])
    target = np.concatenate(target.values) # align type

    # iteration for numerical optimization
    for step in range(0,num_steps):
        # update weights value at each loop
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal) # gradient: derivative of the log_likelihood
        weights += learning_rate * gradient

    print(predictions)
    return weights # find the param to max ll


logistic_regression(features, ins,
                    num_steps = 80000, # 90000 100000 200000
                    learning_rate = 0.0001)

###################################################################
########################   statsmodels    #########################
###################################################################
model1 = sm.Logit(ins, features)
r = model1.fit()
print(r.summary())

###################################################################
##########################   sklearn    ###########################
###################################################################
model2 = LogisticRegression()
model2.fit(df[['age']].values, df[['has_insurance']].values)
model2.intercept_
model2.coef_
