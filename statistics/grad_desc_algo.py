import numpy as np
import matplotlib.pyplot as plt

# basic gradient descendent
def gradient_descent(gradient, start, learn_rate, n_iter, tolerance):
    x = start
    steps = [x]
    for i in range(n_iter):
        x -= gradient(x) * learn_rate # share of the derivative information used
        if abs(x) <= tolerance:
            break
        steps.append(x)
    print(f'Minimum reached in {i} iterations')
    return steps

xsteps = gradient_descent(
    gradient=lambda x: 2*x, # 2x derivative of f(x) = x^2 to minimize
    start=10.0,
    learn_rate=0.1,
    n_iter=100,
    tolerance=0.000001
)

xsteps
'{:.10f}'.format(xsteps[-1]) # numeric solution very close to zero

s = np.array(xsteps)
x = np.linspace(-10, 10)

fig, ax = plt.subplots()
ax.plot(x, x**2, linewidth=2)
ax.plot(s, s**2, '--bo')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()


# bivariate gradient descendent
'''
# function
f = x**2 + y**2

# First partial derivative with respect to x
fpx = 2*x

# First partial derivative with respect to y
fpy = 2*y

# Gradient
grad = [fpx, fpy]
'''

xsteps = gradient_descent(
    gradient=lambda x: 2 * x, # partial derivatives of function f(x) = x^2 + y^2 to minimize
    start=10.0,
    learn_rate=0.2,
    n_iter=50,
    tolerance=1e-06
)

ysteps = gradient_descent(
    gradient=lambda y: 2 * y, # partial derivatives of function f(x) = x^2 + y^2 to minimize
    start=10.0,
    learn_rate=0.2,
    n_iter=50,
    tolerance=1e-06
)

sx = np.array(xsteps)
x = np.linspace(-10, 10)

sy = np.array(ysteps)
y = np.linspace(-10, 10)

x, y = np.meshgrid(x, y)
z = np.array(x**2 + y**2)

sx, sy = np.meshgrid(sx, sy)
sz = np.array(sx**2 + sy**2)
sz

fig = plt.figure()
fig.suptitle('3D  Gradient Descent Algorithm', fontsize=14)
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z, alpha=0.4)
ax.plot_surface(sx, sy, sz, alpha=1)
plt.show()


############################################################
################## learning rate impact ####################
############################################################
lsteps = gradient_descent(
    gradient=lambda x: 2 * x,
    start=10.0,
    learn_rate=0.9, # larger steps => lower accuracy and local min trap
    n_iter=50,
    tolerance=1e-06
)

'{:.10f}'.format(lsteps[-1])
y = np.array(lsteps)
x = np.linspace(-10, 10)

fig, ax = plt.subplots()
ax.plot(x, x**2, linewidth=2)
ax.plot(y, y**2, '--bo')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()

lsteps = gradient_descent(
    gradient=lambda x: 2 * x,
    start=10.0,
    learn_rate=0.005, # smaller steps => higher accuracy but stops before max
    n_iter=50,
    tolerance=1e-06
)

'{:.10f}'.format(lsteps[-1])
y = np.array(lsteps)
x = np.linspace(-10, 10)

fig, ax = plt.subplots()
ax.plot(x, x**2, linewidth=2)
ax.plot(y, y**2, '--bo')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()


############################################################
################## n iterations impact #####################
############################################################
lsteps = gradient_descent(
    gradient=lambda x: 2 * x,
    start=10.0,
    learn_rate=0.005,
    n_iter=1000, # more iter => slower convergence but higher accuracy
    tolerance=1e-06
)

'{:.10f}'.format(lsteps[-1])
y = np.array(lsteps)
x = np.linspace(-10, 10)

fig, ax = plt.subplots()
ax.plot(x, x**2, linewidth=2)
ax.plot(y, y**2, '--bo')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()


############################################################
############### local max-min and saddle ###################
############################################################
lsteps = gradient_descent(
    gradient=lambda x: 4 * x**3 - 10 * x - 3,  # f(x) = x^4 - 5x^2 - 3x
    start=0,        # algorithm trapped in local min:
    learn_rate=0.2, # learning rate or starting point make the difference between finding a local minimum and finding the global minimum
    n_iter=50,
    tolerance=1e-06
)

'{:.10f}'.format(lsteps[-1])
y = np.array(lsteps)
x = np.linspace(-3, 3)

fig, ax = plt.subplots()
ax.plot(x, x**4 - 5*x**2 - 3*x, linewidth=2)
ax.plot(y, y**4 - 5*y**2 - 3*y, '--bo')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()


lsteps = gradient_descent(
    gradient=lambda x: 4 * x**3 - 10 * x - 3,  # f(x) = x^4 - 5x^2 - 3x
    start=0,
    learn_rate=0.1, # get global minimum
    n_iter=50,
    tolerance=1e-06
)

'{:.10f}'.format(lsteps[-1])
y = np.array(lsteps)
x = np.linspace(-3, 3)

fig, ax = plt.subplots()
ax.plot(x, x**4 - 5*x**2 - 3*x, linewidth=2)
ax.plot(y, y**4 - 5*y**2 - 3*y, '--bo')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()


############################################################
##################### gradient matrix ######################
############################################################

def mse_loss(weights, X, Y):
    n = Y.shape[0]
    residuals = np.dot(X, weights) - Y              # e = yp - y
    squared_error = np.dot(residuals.T, residuals)  # s = e'e
    return residuals, (1/n) * squared_error         # mse = 1/n * s

def gradient_descent(X, Y, iterations=100, learn_rate=0.01):
    n = X.shape[0] # obs
    m = X.shape[1] # cols
    weights = np.zeros((m, 1)) # mx1
    losses = []

    for i in range(iterations):
        residuals, loss = mse_loss(weights, X, Y)   # nx1
        gradient = (2/n) * np.dot(residuals.T, X).T # mx1 (1xn x nxm)'
        weights = weights - (learn_rate * gradient) # mx1 (mx1 - mx1)
        losses.append(loss)
        # print(f"Iter: {i} | Cost: {loss} | Weights: {weights}")

    return weights


# data
y = np.array([[31, 28, 29, 31, 23, 27, 30, 24, 32, 31, 24, 33, 33, 35, 37, 31, 27, 34, 39, 25]]).T
y

X = np.array([(90, 178), (72, 180), (48, 161), (90, 176), (48, 164), (76, 190), (62, 175), (52, 161), (93, 190), (72, 164),
              (70, 178), (60, 167), (61, 178), (73, 180), (70, 185), (89, 178), (68, 174), (72, 173), (85, 184), (76, 168)])
X = np.insert(X, 0, 1, axis=1)
X

w = gradient_descent(X, y, 500, 0.00001)
w[0]
w[1]
w[2]


# check ols
inv = np.linalg.inv(np.dot(X.T, X)) # (X'X)^-1
xy = np.dot(X.T, y) # X'y

b = np.dot(inv, xy)
b # b = (X'X)^-1 * X'y


# plot
ax = plt.axes(projection='3d')
p = ax.scatter(X[:, 1], X[:, 2], y, alpha=0.8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
# surface data
x1s = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 1)
x2s = np.arange(np.min(X[:, 2]), np.max(X[:, 2]), 1)
x1s, x2s = np.meshgrid(x1s, x2s)
ys = w[0] + w[1]*x1s + w[2]*x2s
ax.plot_surface(x1s, x2s, ys, alpha=0.4, color='red') # surface
plt.show()

############################################################
##################### gradient loops #######################
############################################################

def predicted_y(weights, X):
    y_pred = []
    n = X.shape[0] # obs
    m = X.shape[1] # cols
    for i in range(n):
        y_pred.append(np.dot(X[i], weights.reshape(m, 1))) # 1xm x mx1
    return y_pred
    # return np.dot(X, weights)

# linear loss
def loss(y, y_predicted):
    s = 0
    n = X.shape[0] # obs
    for i in range(n):
        s += (y[i]-y_predicted[i])**2
    return (1/n) * s

# derivative of loss on weights
def dldw(X, y, y_predicted):
    s = 0
    n = X.shape[0] # obs
    for i in range(n):
        s += -X[i] * (y[i] - y_predicted[i])
    return (2/n) * s

# gradient function
def gradient_descent(X, y, learn_rate, n_iter):
    m = X.shape[1] # cols
    weights_vector = np.zeros(m) # np.zeros((m, 1))
    linear_loss = []

    for i in range(n_iter):
        y_predicted = predicted_y(weights_vector, X)
        weights_vector = weights_vector - learn_rate * dldw(X, y, y_predicted)
        linear_loss.append(loss(y, y_predicted))

    plt.plot(np.arange(1, n_iter+1), linear_loss[0:])
    plt.xlabel("number of epoch")
    plt.ylabel("loss")
    plt.show()

    return weights_vector


# data
y = np.array([[31, 28, 29, 31, 23, 27, 30, 24, 32, 31, 24, 33, 33, 35, 37, 31, 27, 34, 39, 25]]).T
y

X = np.array([(90, 178), (72, 180), (48, 161), (90, 176), (48, 164), (76, 190), (62, 175), (52, 161), (93, 190), (72, 164),
              (70, 178), (60, 167), (61, 178), (73, 180), (70, 185), (89, 178), (68, 174), (72, 173), (85, 184), (76, 168)])
X = np.insert(X, 0, 1, axis=1)
X

gradient_descent(X, y, learn_rate=0.00001, n_iter=5)

# check ols
inv = np.linalg.inv(np.dot(X.T, X)) # (X'X)^-1
xy = np.dot(X.T, y) # X'y

b = np.dot(inv, xy)
b # b = (X'X)^-1 * X'y
