import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

###########################################################
### f(x) = x**2 => Integral (x**3)/3 => Derivative 2x #####
###########################################################

def f(x):
   return x**2

### integral
x = np.arange(0,5,0.01)
len(x)
x = np.arange(0,5,0.001) # the smaller the step, the better the proxy
len(x)

IntSum = 0
for i in range(0,5000):
    IntSum = IntSum + f(x[i]) * (x[i] - x[i-1]) # b*h => f(x) * (x-xo)
print(IntSum)

(x ** 3)/3
(5**3)/3


x = np.array([0,1,2,3,4,5])
y1 = f(x)

I1 = simps(y1, x)
print(I1)

### derivative
x = np.arange(0,6,1)
dydx = []
for i in range(1,6):
    dy = (f(x[i]) - f(x[i-1]))
    dx = (x[i] - x[i-1])
    dydx.append(dy/dx) # derivata
    print(f'x: {x[i]} - {x[i-1]}, dx: {dx}, y: {f(x[i])} - {f(x[i-1])}, dy: {dy}')


x = np.arange(0,5,1)
dydx = np.array(dydx)
dydx

for i in range(1,5):
    dy = (dydx[i]- dydx[i-1])
    dx = (x[i] - x[i-1])
    print(f'x: {x[i]} - {x[i-1]}, dx: {dx}, y: {dydx[i]} - {dydx[i-1]}, dy: {dy}')



def derive(function, value):
    h = 0.001
    dy = function(value + h) - function(value)
    dx = h
    slope = dy / dx
    return slope

derive(f,2)

a = 2.001 ** 2
a
b = 2 ** 2
b
a-b
(a-b)/0.001

### partial derivative

'''
# function
f = x**2 + y**2 - 2*x*y

# First partial derivative with respect to x
fpx = 2*x - 2*y

# First partial derivative with respect to y
fpy = 2*y - 2*x

# Gradient
grad = [fpx, fpy]
'''

def fun(x, y):
    return x**2 + y**2
    #return -np.exp(x**2+y**2) # -2.718281**(x**2+y**2)

x = np.linspace(-0.8, 0.8, 100)
y = np.linspace(-0.6, 0.6, 100)
x, y = np.meshgrid(x, y)
z = np.array(fun(x, y))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z)
plt.show()
