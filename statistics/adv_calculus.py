import io
import requests
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from scipy.misc import derivative

##########################################
### Analytical Case: f formula is given ##
##########################################

# define all symbols in sympy
x, a, b, c = smp.symbols('x a b c', real=True)
x**2+smp.exp(a)

x, a, b, c = smp.symbols('x a b c', real=True)
f = smp.exp(-a*smp.sin(x**2)) * smp.sin(b**x) * smp.log(c*smp.sin(x)**2 / x)
f

# compute derivatives using smp.diff(f, x) where f is the function to derive respect to variable x
dfdx = smp.diff(f, x)
dfdx

# compute the nth derivative d^n f/dx^n by putting the optional argument at the end smp.diff(f,x,n)
d4fdx4 = smp.diff(f, x, 4)
d4fdx4

# compute numerical result by plugging in numbers
d4fdx4.subs([(x, 4), (a, 1), (b, 2), (c, 3)]).evalf()

# convert to a numerical function for plotting (data is discrete!)
d4fdx4_f = smp.lambdify((x, a, b, c), d4fdx4)

# define x and y arrays using the numerical function
x = np.linspace(1, 2, 100)
y = d4fdx4_f(x, a=1, b=2, c=3)

# plot function
plt.plot(x, y)
plt.ylabel('d^4 f / dx^4', fontsize=20)
plt.xlabel('x', fontsize=20)


########################################
#### Numerical Case: f data is given ###
########################################

# WARNING: works fine if the data is smooth but not if the data is noisy
smooth_data = requests.get('https://raw.githubusercontent.com/rpalloni/dataset/master/sample_data1.txt')
x, y = np.loadtxt(io.BytesIO(smooth_data.content))
plt.plot(x, y, 'o--')

dydx = np.gradient(y, x) # calculate derivative (rate of change)
dydx
# WARNING: gradient() applies first difference at beginning and end of series and double difference in the middle
# dy/dx: (0.02-0)/(0.52-0), (0.11-0)/(1.05-0), (0.24-0.02)/(1.57-0.52) ...

plt.plot(x, y, 'o--', label='y(x)')
plt.plot(x, dydx, 'o--', label='y\'(x)')
plt.legend()


noisy_data = requests.get('https://raw.githubusercontent.com/rpalloni/dataset/master/sample_data2.txt')
x, y = np.loadtxt(io.BytesIO(noisy_data.content))
plt.plot(x, y, 'o--')

dydx = np.gradient(y, x)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(x, y, label='y(x)')
ax[1].plot(x, dydx, label='y\'(x)', color='r')
[a.legend() for a in ax]
plt.show() # Noise gets amplified in the derivative!!!! => need to smooth data


noisy_covid_data = requests.get('https://raw.githubusercontent.com/rpalloni/dataset/master/coviddata.txt')
x, y = np.loadtxt(io.BytesIO(noisy_covid_data.content))
plt.plot(x, y)

dydx = np.gradient(y, x)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(x, y, label='y(x)')
ax[1].plot(x, dydx, label='y\'(x)', color='r')
[a.legend() for a in ax]
plt.show() # amplified noise in derivative: no change visible


# smoothing data convolving it with a rectangle: get the moving average of data subsets
filt = np.ones(15)/15

y_smooth = np.convolve(y, filt, mode='valid')
dysdx = np.gradient(y_smooth, x[7:-7])

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(x, y, label='$y(x)$')
ax[0].plot(x[7:-7], y_smooth, label=r'$y_{{smooth}}(x)$')
ax[1].plot(x, dydx, label='$y\'(x)$', color='r')
ax[1].plot(x[7:-7], dysdx, label='$y_{smooth}\'(x)$', color='purple')
ax[1].set_ylim(-100, 120)
ax[1].grid()
[a.legend() for a in ax]
[a.set_xlabel('Time [Days]') for a in ax]
ax[0].set_ylabel('Cases per Day')
ax[1].set_ylabel('$\Delta$ (Cases per Day) / $\Delta t$')
fig.tight_layout()
plt.show()


########################################
########### Function and Data ##########
########################################

def f(u):
    return max(np.abs(np.exp(-x*u**2) - y))

x = np.linspace(0, 1, 500)
y = np.exp(-x*2.15**2) + 0.1*np.random.randn(len(x))
u = np.linspace(0, 10, 40)

plt.scatter(x, y)
plt.xlabel('$x_i$', fontsize=20)
plt.ylabel('$y_i$', fontsize=20)
plt.show()

f_u = np.vectorize(f)(u)
f_u

plt.plot(u, f_u)
plt.xlabel('$u$', fontsize=20)
plt.ylabel('$f(u)$', fontsize=20)
plt.show()

derivative(f, 1, dx=1e-6)

dfdu = np.vectorize(derivative)(f, u, dx=1e-6)

plt.plot(u, dfdu)
