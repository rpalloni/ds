'''
Principal Component Analysis is a mathematical technique used for dimensionality reduction.
Its goal is to reduce the number of features while keeping most of the original information.
Singular Value Decomposition
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

# two vars and ten samples
X = np.array([(8, 14), (11, 24), (10, 18), (13, 20), (9, 19), (5, 9), (2, 7), (1, 5), (3, 3), (4, 8)], dtype=float)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# average of vars
mu1, mu2 = np.mean(X, axis=0)

plt.scatter(X[:, 0], X[:, 1])
plt.title('vars mean on axes')
plt.plot(mu1, 0, 'rs')
plt.plot(0, mu2, 'gs')
plt.axis([0, 15, 0, 30])
for i in range(len(X)):
    plt.arrow(X[i, 0], X[i, 1], -1, 0, width=0.01, length_includes_head=True, head_width=0.5, head_length=0.2, fill=False, color='green')
    plt.arrow(X[i, 0], X[i, 1], 0, -3, width=0.01, length_includes_head=True, head_width=0.2, head_length=0.5, fill=False, color='red')
plt.annotate('X', xy=(mu1, mu2), weight='bold') # center of data
plt.show()


# shift center of data to origin (scale) and add PCA line
X1 = (X[:, 0] - mu1).reshape(10, 1)
X2 = (X[:, 1] - mu2).reshape(10, 1)

inv = np.linalg.inv(np.dot(X1.T, X1)) # (X'X)^-1
xy = np.dot(X1.T, X2) # X'y
m = np.dot(inv, xy) # b = (X'X)^-1 * X'y

plt.scatter(X1, X2)
plt.plot(X1, X1*m, '-', color='g') # regression line
plt.title('X in origin')
plt.annotate('X', xy=(np.mean(X1), np.mean(X2)), weight='bold') # center of data
plt.show()
