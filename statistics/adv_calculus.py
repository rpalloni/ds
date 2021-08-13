import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Symbolic Case

# Define all your symbols in sympy
x, a, b, c = smp.symbols('x a b c', real=True)
x**2+smp.exp(a)

x, a, b, c = smp.symbols('x a b c', real=True)
f = smp.exp(-a*smp.sin(x**2)) * smp.sin(b**x) * smp.log(c*smp.sin(x)**2 /x)
f

# Compute derivatives using smp.diff(f, x) where $f$ is the function you want to take the derivative of
# and $x$ is the variable you are taking the derivative with respect to.
dfdx = smp.diff(f, x)
dfdx

# Can take the nth derivative $d^n f/dx^n$ by putting the optional argument at the end smp.diff(f,x,n)
d4fdx4 = smp.diff(f, x, 4)
d4fdx4

d4fdx4.subs([(x,4),(a,1),(b,2),(c,3)]).evalf()


d4fdx4_f = smp.lambdify((x,a,b,c), d4fdx4)

x = np.linspace(1,2,100)
y = d4fdx4_f(x, a=1, b=2, c=3)


plt.plot(x,y)
plt.ylabel('$d^4 f / dx^4$', fontsize=24)
plt.xlabel('$x$', fontsize=24)
