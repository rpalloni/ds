import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer

pd.options.display.max_columns = None


# data: https://www.kaggle.com/cdc/national-health-and-nutrition-examination-survey

demographic_df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/nutrition/demographic.csv', encoding='latin-1').set_index('SEQN') # person sequence number
demographic_df.shape
demographic_df.head()

examination_df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/nutrition/examination.csv', encoding='latin-1').set_index('SEQN')
examination_df.shape
examination_df.head()

diet_df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/nutrition/diet.csv', encoding='latin-1').set_index('SEQN')
diet_df.shape
diet_df.head()

labs_df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/nutrition/labs.csv', encoding='latin-1').set_index('SEQN')
labs_df.shape
labs_df.head()

# medications_df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/nutrition/medications.csv', encoding='latin-1').set_index('SEQN')
# medications_df.shape
# medications_df.head()

questionnaire_df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/nutrition/questionnaire.csv', encoding='latin-1').set_index('SEQN')
questionnaire_df.shape
questionnaire_df.head()

# merge dataset
df = pd.concat([
    demographic_df,
    examination_df,
    diet_df,
    labs_df,
    questionnaire_df
], axis=1)

df.head()
df.shape

####################################################################################################
######################################## data cleaning #############################################
####################################################################################################

DROP_COLS = []
WHITELIST_COLS = []

# nulls
nulls = pd.isna(df).sum() / len(df) * 100
nulls.sort_values(ascending=False)
nulls.plot(kind="hist", xticks=range(0, 110, 10))
plt.show()
DROP_COLS += nulls[nulls > 50].index.tolist() # drop cols with more than 50% of cells null

# low cardinality (constant/almost constant values in the column)
uniq = df.nunique()
uniq.sort_values()
DROP_COLS += uniq[uniq <= 1].index.tolist() # drop cols with only 0 or 1 values


# demographics
df[['RIDAGEYR', 'RIDAGEMN']].hist() # age yrs and age months
DROP_COLS += ['RIDAGEMN']
WHITELIST_COLS += ['RIDAGEYR']

df['RIAGENDR'].value_counts() # M/F 50%
WHITELIST_COLS += ['RIAGENDR']

df.hist('INDHHIN2', figsize=(8, 6), color='red', bins=50) # skewed hh income
plt.show()
WHITELIST_COLS += ['INDHHIN2']

# examinations
oe = OrdinalEncoder()
oe.fit_transform(df[[x for x in df.columns if x.startswith('OHX')]]) # tooth caries (categorical to numerical)
WHITELIST_COLS += [x for x in df.columns if x.startswith('OHX')]

df['SMD100BR'].value_counts() # cigarettes brand
ft = FunctionTransformer(lambda x: pd.notna(x).astype(int))
ft.fit_transform(df['SMD100BR']).value_counts() # smoker/non-smoker (categorical to dummy)
DROP_COLS += ['SMDUPCA', 'CSXTSEQ'] # drop additional smokers info
WHITELIST_COLS += ['SMD100BR']


DROP_COLS = [x for x in set(DROP_COLS) if x not in WHITELIST_COLS]
df = df.drop(columns=DROP_COLS)
df.shape


####################################################################################################
############################################# PCA ##################################################
####################################################################################################
