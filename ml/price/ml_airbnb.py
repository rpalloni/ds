# price definition model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# airbnb data London
# http://insideairbnb.com/get-the-data.html

listings_file_path = 'http://data.insideairbnb.com/united-kingdom/england/london/2021-12-07/data/listings.csv.gz'
listings = pd.read_csv(listings_file_path, compression='gzip', low_memory=False)
listings.columns
listings.shape

listings.head()

data = listings[['neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates',
                 'bathrooms', 'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',
                 'minimum_minimum_nights', 'maximum_minimum_nights', 'maximum_maximum_nights', 'has_availability',
                 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating']]
data.head()
data.info()

################################################################################################
####################################### preprocessing ##########################################
################################################################################################

# nulls
percentage_missing_data = data.isnull().sum() / data.shape[0]
ax = percentage_missing_data.plot(kind='bar', color='#E35A5C', figsize=(16, 5))
ax.set_xlabel('Feature')
ax.set_ylabel('Percent Empty / NaN')
ax.set_title('Features Emptiness')
plt.show()

# low cardinality
uniq = data.nunique()
uniq.sort_values()
uniq[uniq <= 1].index.tolist()

# eda
count_per_ngc = data['neighbourhood_cleansed'].value_counts()
ax = count_per_ngc.plot(kind='bar', color='#E35A5C', alpha=0.85, figsize=(16, 5))
ax.set_title("Neighborhoods by Number of Listings")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("# of Listings")
plt.show()

data['property_type'].value_counts()
data['property_type'].value_counts().loc[lambda x: x > 10]
data['room_type'].value_counts()
data['accommodates'].value_counts()
data['bathrooms'].value_counts(dropna=False) # remove
data = data.drop(['bathrooms'], axis=1)
data['bathrooms_text'].value_counts()
data['bedrooms'].value_counts(dropna=False) # NaN
data['beds'].value_counts(dropna=False) # NaN

# price
data['price'] = pd.to_numeric(data['price'].str.replace('\$|,', ''))
data['price'].describe()
data['price'].hist(bins=50)
plt.show()

# amenities
data['amenities'] = data['amenities'].apply(eval) # string to list

def get_amenities(series):
    return pd.Series([x for ls in series for x in ls])

get_amenities(data['amenities']).nunique()
get_amenities(data['amenities']).unique()
get_amenities(data['amenities']).value_counts()[0:50]

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(get_amenities(data['amenities']).value_counts()[0:50].index,
       get_amenities(data['amenities']).value_counts()[0:50].values,
       color='#E35A5C', )
ax.set_ylabel('Frequency', size=12)
ax.tick_params(axis='x', rotation=90)
ax.set_title('Most frequent amenities', size=14)
plt.show()

def dummify_amenities(item_lists, unique_items):
    bool_dict = {}
    for i, item in enumerate(unique_items):
        bool_dict[item] = item_lists.apply(lambda x: item in x)
    return pd.DataFrame(bool_dict)

d_amenities = dummify_amenities(data['amenities'], get_amenities(data['amenities']).unique())
d_amenities

data = pd.concat([data, d_amenities], axis=1)
data.shape

data.groupby(['Wifi', ]).agg({'price': 'median'})


data['has_availability'].value_counts() # int64
data = data.drop(['has_availability'], axis=1)

data['number_of_reviews'].describe()
data['review_scores_rating'].describe()


# features correlations: multicollinearity (remove nulls and encode categoricals)
temp_data = data.copy()
temp_data = temp_data.dropna(axis=0)

def encode_categorical(array):
    if not array.dtype == np.dtype('float64'):
        return OrdinalEncoder().fit_transform(array)
    else:
        return array
temp_data = temp_data.apply(encode_categorical)

corr_matrix = temp_data.corr()
corr_matrix

# heat map
sns.heatmap(corr_matrix, cmap='Blues',
            xticklabels=corr_matrix.columns,
            yticklabels=corr_matrix.columns)
plt.show()


################################################################################################
########################################## modeling ############################################
################################################################################################

# Shuffle the data to ensure a good distribution for the training and testing sets
df = shuffle(data)

# Extract features and labels
y = df['price']
X = df.drop('price', axis=1)

# Training and Testing Sets
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=123)

train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)

train_X.shape, test_X.shape


def get_model(df):

    cat_df = df.select_dtypes(include=[object])
    ct = ColumnTransformer([
        ('ordinal_encoder', OrdinalEncoder(), [df.columns.tolist().index(x) for x in cat_df.columns]),
    ], remainder='passthrough')

    si = SimpleImputer(strategy='most_frequent')

    ss = StandardScaler()

    estimator = RandomForestRegressor(random_state=123)

    pipe = Pipeline([
        ('column_transformer', ct),
        ('imputer', si),
        ('scaler', ss),
        ('estimator', estimator)
    ])

    return pipe


def evaluate_model(model, predict_set, evaluate_set):
    predictions = model.predict(predict_set)
    print("Median Absolute Error: " + str(round(median_absolute_error(predictions, evaluate_set), 2)))
    RMSE = round(np.sqrt(mean_squared_error(predictions, evaluate_set)), 2)
    print("RMSE: " + str(RMSE))


# DAG style steps representation
pipe = get_model(df[0:999])
set_config(display='diagram')
pipe

pipe.fit(train_X, train_y)
evaluate_model(pipe, test_X, test_y)
