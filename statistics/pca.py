'''
Principal Component Analysis is a mathematical technique used for dimensionality reduction.
Its goal is to reduce the number of features while keeping most of the original information.
Singular Value Decomposition
'''

import numpy as np
import matplotlib.pyplot as plt

# two vars and ten samples
X = np.array([(10, 21), (11, 23), (8, 25), (13, 20), (9, 19), (5, 9), (2, 7), (1, 5), (3, 6), (4, 8)], dtype=float)
X

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
X[:, 0] = X[:, 0] - mu1
X[:, 1] = X[:, 1] - mu2

mu1, mu2 = np.mean(X, axis=0)
d = 0.5

pcax = np.linspace(min(X[:, 0]), max(X[:, 0]), len(X))
pcay = np.linspace(min(X[:, 1]), max(X[:, 1]), len(X))

plt.scatter(X[:, 0], X[:, 1])
plt.title('X in origin')
plt.axis([min(X[:, 0])-d, max(X[:, 0])+d, min(X[:, 1])-d, max(X[:, 1])+d])
plt.annotate('X', xy=(mu1, mu2), weight='bold') # center of data
plt.plot(pcax, pcay)
plt.show()
