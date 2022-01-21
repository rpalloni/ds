import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# basic gradient descendent
def gradient_descent(gradient, start, learn_rate, n_iter, tolerance):
    vector = start
    steps = [vector]
    for i in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if abs(diff) <= tolerance:
            break
        vector += diff
        steps.append(vector)
    return steps

xsteps = gradient_descent(
    gradient=lambda x: 2*x, # 2x derivative of f(x) = x^2 to minimize
    start=10.0,
    learn_rate=0.2,
    n_iter=50,
    tolerance=1e-06
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
    learn_rate=0.8, # larger steps
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
####################### ols gradient #######################
############################################################

def predicted_y(weights, X, intercept):
    y_pred = []
    for i in range(len(X)):
        y_pred.append(weights@X[i]+intercept)
    return np.array(y_pred)

# linear loss
def loss(y, y_predicted):
    s = 0
    n = len(y)
    for i in range(n):
        s += (1/n) * (y[i]-y_predicted[i])**2
    return s

# derivative of loss on weights
def dldw(X, y, y_predicted):
    s = 0
    n = len(y)
    for i in range(n):
        s += (2/n) * -X[i] * (y[i] - y_predicted[i])
    return s

# derivative of loss on intercept (bias)
def dldb(y, y_predicted):
    s = 0
    n = len(y)
    for i in range(len(y)):
        s += (2/n) * -(y[i] - y_predicted[i])
    return s

# gradient function
def gradient_descent(X, y, learn_rate=0.001, n_iter=2000):
    weights_vector = [0, 0]
    intercept = 0
    linear_loss = []

    for i in range(n_iter):
        y_predicted = predicted_y(weights_vector, X, intercept)
        weights_vector = weights_vector - learn_rate * dldw(X, y, y_predicted)
        intercept = intercept - learn_rate * dldb(y, y_predicted)
        linear_loss.append(loss(y, y_predicted))

    plt.plot(np.arange(1, n_iter), linear_loss[1:])
    plt.xlabel("number of epoch")
    plt.ylabel("loss")
    plt.show()

    return weights_vector, intercept
