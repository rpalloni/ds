import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # using Pipeline to abstract training steps
from sklearn.preprocessing import LabelBinarizer, LabelEncoder # transformers for target
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder # transformers for features
from sklearn.tree import DecisionTreeClassifier, plot_tree # estimator
from sklearn.svm import SVC # estimator
from sklearn.multiclass import OneVsRestClassifier # estimator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score # metrics

iris = datasets.load_iris() # classes: 0-setosa, 1-virginica, 2-versicolor

iris # dict object with data and metadata

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target_num'] = iris.target
df['target_class'] = df['target_num'].apply(lambda x: iris.target_names[x])
df['target_is_setosa'] = (df['target_class'] == 'setosa').astype(int)

df.head()
df.describe()
df['target_class'].value_counts()
df['target_class'].value_counts().plot(kind='barh', color=['green','purple','blue'])

sns.pairplot(df, hue='target_class')
plt.show()

# train test
X = df[iris.feature_names]
y = df['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=123)
y.value_counts() / len(y)

# dummify
y_dummy = df['target_is_setosa']

y_dummy.value_counts() / len(y_dummy) # unbalanced classes

X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X, y_dummy,
                                                                            test_size=0.25, # 25% data in test
                                                                            shuffle=True, # mix rows
                                                                            stratify=y_dummy, # keep df shares
                                                                            random_state=123) # seed
y_train_dummy.value_counts() / len(y_train_dummy)
y_test_dummy.value_counts() / len(y_test_dummy)


# encode multiclass string to number
le = LabelEncoder()
y_train_le = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_le = pd.Series(le.transform(y_test), index=y_test.index)

le.classes_
le.inverse_transform([0, 1, 2])

# normalize features to remove scale bias due to features magnitude
ss = StandardScaler()
ss.fit(X_train)

ss.mean_
ss.scale_

X_train_ss = pd.DataFrame(ss.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_ss = pd.DataFrame(ss.transform(X_test), columns=X_test.columns, index=X_test.index)

# Binary Decision Tree
dtc = DecisionTreeClassifier(max_depth=2, random_state=123)
dtc.fit(X_train_dummy, y_train_dummy)

plot_tree(dtc)
plt.show() # binary tree rule: if petal length <= 2.6 than setosa
X.columns[2]

y_pred_dummy = dtc.predict(X_test_dummy)
y_pred_dummy


# Multiclass
ovr = OneVsRestClassifier(estimator=SVC(random_state=123))
ovr.fit(X_train_ss, y_train_le)

y_pred = ovr.predict(X_test_ss)


# Use Pipeline
pipe = Pipeline(steps=[
    ('scaler', ss),          # first pipeline step
    ('over_classifier', ovr) # second pipeline step
])

pipe.fit(X_train, y_train_le)
pipe.score(X_test, y_test_le) # accuracy
y_pred_pipe = pipe.predict(X_test)
y_pred_pipe

pipe.get_params() # pipe params for transformer/estimator config
pipe.set_params(over_classifier__estimator__kernel='linear') # use linear kernel
pipe.set_params(scaler='passthrough') # skip scaling step


# Result assessment
# true positive
# true negative
# false positive
# false negative

cm = confusion_matrix(y_test_le, y_pred_pipe)
cm # main diagonal ok

print(classification_report(y_test_le, y_pred_pipe))

# accuracy: true positive + true negative / tot
accuracy_score(y_test_le, y_pred_pipe)
# precision: true positive / true positive + false positive
precision_score(y_test_le, y_pred_pipe, average='weighted')
# recall: true positive / true positive + false negative
recall_score(y_test_le, y_pred_pipe, average='weighted')
