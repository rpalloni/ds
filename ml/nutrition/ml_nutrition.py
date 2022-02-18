import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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
df.dtypes.value_counts() # presence of object cols (categorical)

####################################################################################################
######################################## data cleaning #############################################
####################################################################################################

DROP_COLS = []

# nulls
nulls = pd.isna(df).sum() / len(df) * 100
nulls.sort_values(ascending=False)
nulls.plot(kind='hist', xticks=range(0, 110, 10))
plt.show()
DROP_COLS += nulls[nulls > 50].index.tolist() # drop cols with more than 50% of cells null

# low cardinality (constant/almost constant values in the column)
uniq = df.nunique()
uniq.sort_values()
DROP_COLS += uniq[uniq <= 1].index.tolist() # drop cols with only 0 or 1 values


# demographics
df[['RIDAGEYR', 'RIDAGEMN']].hist() # age yrs and age months
DROP_COLS += ['RIDAGEMN']

df['RIAGENDR'].value_counts() # M/F 50%

df.hist('INDHHIN2', figsize=(8, 6), color='red', bins=50) # skewed hh income
plt.show()

# medical info
categ = df.select_dtypes(include=[object])
[x for x in categ.columns if x.startswith('OHX')] # tooth caries
set(categ) - set([x for x in categ.columns if x.startswith('OHX')]) # smokers info

df['SMD100BR'].value_counts() # cigarettes brand
ft = FunctionTransformer(lambda x: pd.notna(x).astype(int))
ft.fit_transform(df['SMD100BR']).value_counts()
df['SMOKER'] = ft.fit_transform(df['SMD100BR']) # smoker/non-smoker (categorical to dummy)
DROP_COLS += ['SMDUPCA', 'CSXTSEQ'] # drop additional smokers info

oe = OrdinalEncoder()
oe.fit_transform(df[[x for x in categ.columns if x.startswith('OHX')]])

df = df.drop(columns=DROP_COLS)
df.shape

####################################################################################################
############################################# PCA ##################################################
####################################################################################################

def get_model(df):

    cat_df = df.select_dtypes(include=[object])
    ct = ColumnTransformer([
        ('ordinal_encoder', OrdinalEncoder(), [df.columns.tolist().index(x) for x in cat_df.columns if x.startswith('OHX')]),
    ], remainder='passthrough')

    si = SimpleImputer(strategy='most_frequent')

    ss = StandardScaler()

    pca = PCA(n_components=50, random_state=123) # n components == n columns => set n_components to keep

    pipe = Pipeline([
        ('column_transformer', ct),
        ('imputer', si),
        ('scaler', ss),
        ('pca', pca)
    ])

    return pipe


# DAG style steps representation
pipe = get_model(df)
set_config(display='diagram')
pipe


# plot explained variances
def plot_pca_variances(pca):
    plt.bar(range(pca.n_components), pca.explained_variance_ratio_, color='black')
    plt.title('Variance explained by each component')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.show()


pipe.fit(df)
plot_pca_variances(pipe['pca'])


pipe = get_model(df)
pipe.set_params(pca__n_components=7) # set threshold when % gets flat
pipe.fit(df)
principal_components = pipe.transform(df)
plot_pca_variances(pipe['pca'])


# reduced df
principal_components_df = pd.DataFrame(
    principal_components,
    index=df.index,
    columns=[
        f'PCA{i+1}' for i in range(principal_components.shape[1])
    ]
)
principal_components_df.head()
principal_components_df.shape

principal_components_df.plot(kind='scatter', x='PCA1', y='PCA2', alpha=.1, color='black')
plt.show()
