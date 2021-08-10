# data management
import pandas as pd
# science kit learn
from sklearn.preprocessing import LabelEncoder # import encoder from sklearn library
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# bf_test = pd.read_csv("https://raw.githubusercontent.com/rpalloni/dataset/master/black-friday/test.csv")
bf_train = pd.read_csv("https://raw.githubusercontent.com/rpalloni/dataset/master/black-friday/train.csv")

bf_test.head(20)
bf_test.shape

bf_train.head(20)
bf_train.shape

bf_train['Stay_In_Current_City_Years'].value_counts() # count number of values per variable range
bf_train.isnull().sum() # calculates n of null values for each category in dataset

b = ['Product_Category_2','Product_Category_3'] # array consisting of 2 columns namely Product_Category_2,Product_Category_3

# fill empty with max value
for i in b:
    exec("bf_train.%s.fillna(bf_train.%s.value_counts().idxmax(), inplace=True)" %(i,i))

bf_train.isnull().sum()

# remove purchase
X = bf_train.drop(["Purchase"], axis=1)
X.head(20)

# encode the data into labels using label encoder for easy computing
LE = LabelEncoder()

# apply encoder to data
X = X.apply(LE.fit_transform)

# convert the data into the numerical form using pandas
# dealing with numeric data would be easier than dealing with Categorical data
X.Gender = pd.to_numeric(X.Gender)
X.Age = pd.to_numeric(X.Age)
X.Occupation = pd.to_numeric(X.Occupation)
X.City_Category = pd.to_numeric(X.City_Category)
X.Stay_In_Current_City_Years = pd.to_numeric(X.Stay_In_Current_City_Years)
X.Marital_Status = pd.to_numeric(X.Marital_Status)
X.Product_Category_1 = pd.to_numeric(X.Product_Category_1)
X.Product_Category_2 = pd.to_numeric(X.Product_Category_2)
X.Product_Category_3 = pd.to_numeric(X.Product_Category_3)

Y = bf_train["Purchase"]

# Standardize features by removing the mean and scaling to unit variance
SS = StandardScaler()

# transform X into numeric representation as machine learning methods operate on matrices of number
Xs = SS.fit_transform(X)

# PCA: transform the initial variables in 4 new variables called principal components
pc = PCA(4)

# PCA to data/fitting data to PCA
principalComponents = pc.fit_transform(X)
pc.explained_variance_ratio_

principalDf = pd.DataFrame(data = principalComponents, columns = ["component 1", "component 2", "component 3", "component 4"])

