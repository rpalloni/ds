import numpy as np
import matplotlib.pyplot as plt

# coordinates of ten points (vectors)
x_values = np.array([1.2, 5.3, 4.0, 7.4, 3.5, 6.5, 8.4, 5.5, 7.5, 2.0], dtype=float).reshape(10, 1)
y_values = np.array([3.2, 5.1, 2.8, 5.6, 1.3, 4.5, 8.2, 2.4, 7.6, 5.0], dtype=float).reshape(10, 1)

# rescale to origin
# x_values = x_values - np.mean(x_values)
# y_values = y_values - np.mean(y_values)

# coordinate of projection vector => regression line
# the position of this vector marks the position of the line
inv = np.linalg.inv(np.dot(x_values.T, x_values)) # (X'X)^-1
xy = np.dot(x_values.T, y_values) # X'y
m = np.dot(inv, xy) # b = (X'X)^-1 * X'y
pv = np.array([1, m[0][0]]).reshape(2, 1) # y = x*m

# coordinates of points orthogonal projections on regression line
def get_projection(v1, v2=pv):
    inv = np.linalg.inv(np.dot(v2.T, v2))
    xy = np.dot(v2.T, v1)
    b = np.dot(inv, xy)
    # print(b)
    proj = v2 * b
    return proj


for i in range(len(x_values)):
    v1 = np.array([x_values[i][0], y_values[i][0]]).reshape(2, 1)
    proj = get_projection(v1) # get projection of each point on the regression line

    plt.scatter(proj[0], proj[1], color='r') # projections
    plt.plot([x_values[i], proj[0]], [y_values[i], proj[1]], alpha=0.3, color='r', linestyle='--')
    plt.plot([x_values[i], x_values[i]], [y_values[i], x_values[i]*m[0][0]], alpha=0.3, color='y', linestyle='--')
plt.scatter(x_values, y_values, color='b') # data
plt.scatter(x_values, x_values*m, color='y') # regression points
plt.plot(x_values, x_values*m, '-', color='g') # regression line
# plt.axvline(x=0)
# plt.axhline(y=0)
plt.show()
