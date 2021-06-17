import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

download_url = "https://raw.githubusercontent.com/rpalloni/dataset/master/winequality.csv"
response = requests.get(download_url)
data = pd.read_csv(io.BytesIO(response.content), dtype={
    'fixed acidity': float,'volatile acidity':float, 'citric acid':float,
    'residual sugar':float, 'chlorides':float, 'free sulfur dioxide':float, 'total sulfur dioxide':float,
    'density':float, 'pH':float, 'sulphates':float, 'alcohol': float, 'quality':int})

data.head()
data.shape

data.isnull().values.any()
data.describe()

data.hist(figsize=(10,10), color='red')
plt.show()


target = data.quality
predictors = data.drop('quality', axis=1)

target.head()
predictors.head()

# good > 5 (1) - bad < 5 (0)
q_dummy = (data.quality > 5).astype(int) # aggregated target value
q_dummy.head()


ax = q_dummy.plot.hist(color='green')
ax.set_title('Wine quality distribution', fontsize=14)
ax.set_xlabel('aggregated target value')

# split data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(predictors, q_dummy, test_size=0.2, random_state=123)

# random forest classifier
rf_clf = RandomForestClassifier(random_state=123)

# k-fold cross validation on training dataset and see mean accuracy score
cv_scores = cross_val_score(rf_clf,X_train, y_train, cv=10, scoring='accuracy')
print(f'The accuracy scores for the iterations are {cv_scores}')
print(f'The mean accuracy score is {cv_scores.mean()}')


# predictions
rf_clf.fit(X_train, y_train)
pred_rf = rf_clf.predict(X_test)

for i in range(0,5):
    print('Actual wine quality is ', y_test.iloc[i], ' and predicted is ', pred_rf[i])


# accuracy, log loss and confusion matrix
print(accuracy_score(y_test, pred_rf))
print(log_loss(y_test, pred_rf))

print(confusion_matrix(y_test, pred_rf)) # 38+29 classification error