'''
Principal Component Analysis is a mathematical technique used for dimensionality reduction.
Its goal is to reduce the number of features while keeping most of the original information.
It is based on Singular Value Decomposition: find eigenvalues and eigenvectors of features:
*eigenvectors of the Covariance matrix are the directions of the axes where there is the most variance
*eigenvalues are the coefficients attached to eigenvectors, which give the amount of variance carried in each component
By ranking your eigenvectors in order of their eigenvalues, highest to lowest, you get the principal components in order of significance
Since each PC correspond to a variable, excluding the least significant removes non relevant variables
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, svd

# two vars and ten samples2020
X = np.array([[14, 8], [22, 12], [17, 10], [19, 7], [21, 9], [9, 5], [10, 3], [6, 1], [4, 3], [12, 4]], dtype=float)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# average of vars
mu1, mu2 = np.mean(X, axis=0)

plt.scatter(X[:, 0], X[:, 1])
plt.plot(mu1, 0, 'rs')
plt.plot(0, mu2, 'gs')
plt.axis([0, 25, 0, 15])
for i in range(len(X)):
    plt.arrow(X[i, 0], X[i, 1], -1, 0, width=0.01, length_includes_head=True, head_width=0.5, head_length=0.2, fill=False, color='green')
    plt.arrow(X[i, 0], X[i, 1], 0, -1, width=0.01, length_includes_head=True, head_width=0.2, head_length=0.5, fill=False, color='red')
plt.annotate('X', xy=(mu1, mu2), weight='bold') # center of data
plt.annotate('mx1', xy=(mu1, 0.1)) # center of data
plt.annotate('mx2', xy=(0.1, mu2)) # center of data
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


# shift center of data to origin (scale) and add PC line
X1 = (X[:, 0] - mu1).reshape(10, 1)
X2 = (X[:, 1] - mu2).reshape(10, 1)

# calculate PC line
inv = np.linalg.inv(np.dot(X1.T, X1)) # (X'X)^-1
xy = np.dot(X1.T, X2) # X'y
m = np.dot(inv, xy) # b = (X'X)^-1 * X'y
m

# X1/X2 ratio
c1 = 5
c2 = c1*m
i = np.sqrt(c1**2 + c2**2) # euclidean distance from origin

plt.scatter(X1, X2)
plt.plot(X1, X1*m, '-', color='g')
plt.annotate('X', xy=(np.mean(X1), np.mean(X2)), weight='bold') # center of data
plt.annotate(f'PC1 (m={round(m[0][0],3)})', xy=(4, 7), color='g', weight='bold')
plt.text(-1, -6, 'PC1: X2/X1 ratio \nData more spread along X1 \nX1 more important to describe data variability',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
plt.plot([0, c1], [0, 0], '-', color='r')
plt.plot([c1, c1], [c2[0][0], 0], '-', color='r')
plt.annotate('i', xy=(2.5, 2), color='r')
plt.annotate('c1', xy=(2.5, -0.6), color='r')
plt.annotate('c2', xy=(5.2, 1.2), color='r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.grid(alpha=0.2)
plt.show()


# PC1 loading scores (scaled => i=1)
ls_pc1_x1 = c1/i
ls_pc1_x2 = c2/i

# eigenvector (singular vector): 1 unit long vector (i)
np.sqrt((ls_pc1_x1)**2 + (ls_pc1_x2)**2)

eigenvector_pc1 = [ls_pc1_x1, ls_pc1_x2]


# PC2: PC1 perpendicular line via origin
ls_pc2_x1 = -c2/i
ls_pc2_x2 = c1/i

eigenvector_pc2 = [ls_pc2_x1, ls_pc2_x2]

plt.scatter(X1, X2)
plt.plot(X1, X1*m, '-', color='g')
plt.plot(X1, X1*-(1/m), '-', color='m') # perpendicular: slope => negative, opposite of m
plt.annotate('X', xy=(np.mean(X1), np.mean(X2)), weight='bold')
plt.annotate(f'PC1 (m={round(m[0][0],3)})', xy=(4, 7), color='g', weight='bold')
plt.annotate('PC2: PC1 perpendicular line via origin', xy=(-5, 10), color='m', weight='bold')
plt.plot([0, -c2[0][0]], [0, 0], '-', color='r')
plt.plot([-c2[0][0], -c2[0][0]], [0, c1], '-', color='r')
plt.annotate('i', xy=(-0.5, 2), color='r')
plt.annotate('c1', xy=(-3.8, 2.2), color='r')
plt.annotate('c2', xy=(-2.5, -0.6), color='r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.grid(alpha=0.2)
plt.show()


# eigenvalues: sum of the squared distances between projected points and origin for each PC
v_pc1 = np.array([1, m[0][0]]).reshape(2, 1) # a vector on PC1 line
v_pc2 = np.array([1, -(1/m[0][0])]).reshape(2, 1) # a vector on PC2 line

i_pc1 = np.sqrt(v_pc1[0]**2 + v_pc1[1]**2)
i_pc2 = np.sqrt(v_pc2[0]**2 + v_pc2[1]**2)

u_pc1 = v_pc1/i_pc1 # rescale to unit vector
# np.dot(u_pc1.T, u_pc1) # check 1
u_pc2 = v_pc2/i_pc2

def get_projection(p, v):
    return np.dot(p.T, v) / np.dot(v.T, v)

eigenvalue_pc1 = 0
eigenvalue_pc2 = 0
for i in range(len(X)):
    p = np.array([X1[i][0], X2[i][0]]).reshape(2, 1)
    proj_pc1 = get_projection(p, u_pc1)
    proj_pc2 = get_projection(p, u_pc2)
    eigenvalue_pc1 += proj_pc1**2
    eigenvalue_pc2 += proj_pc2**2

print(eigenvalue_pc1, eigenvalue_pc2)

eigenvalue_pc1 / (len(X) - 1) # PC variation

# PC1_VAR / (PC1_VAR + PC2_VAR) => PC1 % on total variation
(eigenvalue_pc1 / (len(X) - 1)) / ((eigenvalue_pc1 / (len(X) - 1)) + (eigenvalue_pc2 / (len(X) - 1))) # PC1 96% tot var

for i in range(len(X)):
    p = np.array([X1[i][0], X2[i][0]]).reshape(2, 1)
    proj_pc1 = u_pc1 * np.dot(p.T, u_pc1) / np.dot(u_pc1.T, u_pc1)
    proj_pc2 = u_pc2 * np.dot(p.T, u_pc2) / np.dot(u_pc2.T, u_pc2)
    plt.scatter(proj_pc1[0], proj_pc1[1], color='g')
    plt.plot([X1[i][0], proj_pc1[0][0]], [X2[i][0], proj_pc1[1][0]], alpha=0.3, color='g', linestyle='--')
    plt.scatter(proj_pc2[0], proj_pc2[1], color='m')
    plt.plot([X1[i][0], proj_pc2[0][0]], [X2[i][0], proj_pc2[1][0]], alpha=0.3, color='m', linestyle='--')
plt.scatter(X1, X2)
plt.plot(X1, X1*m, '-', color='g')
plt.plot(X1, X1*-(1/m), '-', color='m')
plt.annotate('X', xy=(np.mean(X1), np.mean(X2)), weight='bold')
plt.annotate('PC1', xy=(10, 3), color='g', weight='bold')
plt.annotate('PC2', xy=(-4, 10), color='m', weight='bold')
plt.text(-10, -10, 'PC1 projections have larger variability (spaced green dots) \nthan PC2 projections (close magenta dots) ',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.grid(alpha=0.2)
plt.show()


# eig svd check with numpy
centered_data = X - np.mean(X, axis=0)
eval, evec = eig(np.dot(centered_data.T, centered_data))
eval
evec

U, val, vec = svd(centered_data, full_matrices=False)
val**2
vec

# cov matrix
np.dot(centered_data.T, centered_data) / (len(X) - 1)

#################################################
########### m == cov(x1,x2) / var(x1) ###########
#################################################

# covariance [2x2]
# | var1  cov  |
# | cov   var2 |
var1, var2 = np.var(X, axis=0, ddof=1) # sum((i-mu1)**2 for i in X[:, 0]) / (len(X)-1)
var1
var2

n = len(X)
cov = sum((X[:, 0][i] - mu1) * (X[:, 1][i] - mu2) for i in range(n)) / (n - 1)
cov

covmat = [[var1, cov], [cov, var2]]
covmat

cov/var1 # m!!!
cov/var2

# calculate the mean of each column
M = np.mean(X, axis=0)
# center columns by subtracting column means
C = X - M
C
# calculate covariance matrix of centered matrix
# V = np.cov(C.T)
V = np.dot(C.T, C) / (len(X) - 1)
V
# eigendecomposition of covariance matrix
values, vectors = eig(V)
vectors
values
values*(len(X) - 1)
# project data
P = vectors.T.dot(C.T)
P.T

values[0] / (values[0] + values[1])
np.var(P[0]) / (np.var(P[0]) + np.var(P[1])) # PC1 96% tot var


plt.scatter(X1, X2, alpha=0.2)
plt.scatter(P[0], P[1])
plt.annotate('PC1', xy=(10, -1), color='g')
plt.annotate('PC2', xy=(1, 10), color='m')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.grid(alpha=0.2)
plt.axhline(y=0, c='g')
plt.axvline(x=0, c='m')
plt.show()
