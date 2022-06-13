import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# lending club dataset kaggle
dttr = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/loan/train.csv')
dtts = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/loan/test.csv')


dttr.head()

dttr['loan_risk'].value_counts()
