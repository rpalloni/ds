import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder # transformers for target
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder # transformers for features
from sklearn.tree import DecisionTreeClassifier, plot_tree # estimator
from sklearn.svm import SVC # estimator
from sklearn.multiclass import OneVsRestClassifier # estimator

iris = datasets.load_iris() # classes: 0-setosa, 1-virginica, 2-versicolor

iris # dict object with data and metadata

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target_num'] = iris.target
df['target_class'] = df['target_num'].apply(lambda x: iris.target_names[x])
df['target_is_setosa'] = (df['target_class'] == 'setosa').astype(int)

df.head()
df.describe()
df['target_class'].value_counts()

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
