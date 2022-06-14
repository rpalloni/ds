import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# lending club dataset kaggle
data = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/loan/train.csv')
data_test = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/loan/test.csv')


data.head()

data['loan_risk'].value_counts()
data['loan_risk'] = data['loan_risk'].replace(['Paid', 'Charged off'], [1, 0])
data_test['loan_risk'] = data_test['loan_risk'].replace(['Paid', 'Charged off'], [1, 0])

data_train, data_val = train_test_split(data, test_size=0.2)

def train_xgboost_pipeline(data_x, data_y, parameters=None):

    if parameters is None:
        parameters = dict(n_estimators=100,
                          max_depth=4,
                          scale_pos_weight=1,
                          learning_rate=0.1)

    # feature preprocessing

    numerical_features = data_x.select_dtypes(include='number').columns
    categorical_features = data_x.select_dtypes(include='object').columns

    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('ord', numerical_transformer, numerical_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    # pipeline preprocessor + XGBoost

    xgb_clf = XGBClassifier(n_estimators=parameters['n_estimators'],
                            max_depth=parameters['max_depth'],
                            scale_pos_weight=parameters['scale_pos_weight'],
                            learning_rate=parameters['learning_rate'],
                            random_state=42,
                            n_jobs=4)

    xgb_pipeline = Pipeline(steps=[('preprocessing', preprocessor),
                                   ('xgb_model', xgb_clf)])

    # Model fit

    xgb_pipeline.fit(data_x, data_y)

    return xgb_pipeline


clf = train_xgboost_pipeline(data_train.drop(['loan_risk'], axis=1), data_train['loan_risk'])

report_val = classification_report(data_val['loan_risk'],
                                   clf.predict(data_val.drop(['loan_risk'], axis=1)),
                                   output_dict=True)

report_test = classification_report(data_test['loan_risk'],
                                    clf.predict(data_test.drop(['loan_risk'], axis=1)),
                                    output_dict=True)

report_val

report_test
