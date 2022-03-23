
#####################################################################
################ numpy http://www.numpy.org/ ########################
#####################################################################
import numpy as np
# vectorized operations and relative scientific tools based on vectors
# e.g. np.mean(), np.std(), np.sin(), np.log(), np.random(), etc
# Python standard to store numerical data -> efficient, fast and clean

help(np)
help(np.mean)

# Matrices in python are just numpy arrays with 2 dimensions.
# Create a matrix by passing the numpy array function a list of lists.
# Each inner list will be taken to be one row of the matrix.

x = [1, 2, 3]
y = [4, 6, 8]

# 1D array
a = np.array(x)
a.shape
type(a)
a

# 2D array
b = np.array([x])
b.shape
b # two square brachets

c = np.array(x, ndmin=2)
c.shape
c

d = np.array([x]).T
d.shape
d

e = np.array(x)[:, None]
e.shape

# explicit shape
f = np.array(x).reshape(1, 3)
f.shape
f

g = np.array(y).reshape(3, 1)
g.shape
g

g-f # pair difference

g*f # pair product

np.dot(g, f) # matrix

f*g

np.dot(f, g) # scalar


# dot vs inner
r = np.array(x)
s = np.array(y)

np.dot(r, s)
np.inner(r, s) # no difference with 1D array

p = np.array([[1, 2], [3, 4]])
q = np.array([[5, 3], [1, 6]])
# difference with matrix
np.dot(p, q) # (7 = 1*5 + 2*1)
np.inner(p, q) # (11 = 1*5 + 2*3)

np.dot(a, b) == np.inner(a, b.T)


mat3x2 = np.array([[1, 2], [3, 4], [5, 6]])
print(mat3x2)

mat2x3 = np.array([[1, 2, 3], [4, 5, 6]])
print(mat2x3)

print(mat3x2 @ mat2x3) # matrix multiplication 3x3

mat3x2.dot(mat2x3) # dot product

print(mat2x3 @ mat3x2) # 2x2

mat2x3.dot(mat3x2)

print(mat2x3 @ mat2x3) # not conformable
mat2x3.dot(mat2x3)


M = np.matrix([[5, 5], [-1, 7]])
M.T*M

V = np.array([[5, 5], [-1, 7]])
V.T*V # KO: object type affects result
np.dot(V.T, V) # OK

# Sequences
t = np.linspace(0, 20, 50)
t

s = np.arange(0, 24, 3)
s

# import
dt = np.loadtxt('data.txt', delimiter=',') # mixed data types not supported
dt

# export
np.savetxt('data/array.txt', s, fmt='%i')
# fmt='%i' integer
# fmt='%10.5f' float rounded to five decimals

#######################################################################
############## pandas https://pandas.pydata.org/ ######################
#######################################################################
import pandas as pd
# data management and analysis
# two main data structures: Series and DataFrame

# Series: monodimensional and monotype indexed array
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s

# DataFrame: bidimensional and multitype matrix
index = pd.date_range('1/1/2020', periods=10)
df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20130102'),
                   'C': pd.Series(np.random.randn(10), dtype='float32', index=index),
                   'D': np.array(np.random.randn(10), dtype='int32'),
                   'E': pd.Categorical(["red", "green", "yellow", "red", "yellow", "yellow", "white", "red", "red", "green"]),
                   'F': np.float64([27.25, 70.35, 68.25, 6.30, 29.65, 8.35, 7.85, 84.25, 32.5, 26.35])},
                  index=index)
df

df.shape # nrows ncols

# cols
df.columns
df['E'] # get series using column index (header)
df[['E']] # get column

# rows
df.index
df.loc['2020-01-10'] # localize using row index
df.loc[:'2020-01-05']
df.loc[df['E'] == 'red'] # boolean indexing
df[df['E'] == 'red']
# df.set_index('column')

df.loc['2020-01-02': '2020-01-05'] # slice indexing (row/col values)
df.loc[['2020-01-02', '2020-01-05'], ['C', 'E']]
df.loc['2020-01-02': '2020-01-05', ['C', 'E']]
df['E'].value_counts()
df.loc[3:6] # error!

df[3:6] # position indexing (row/col integer indexes)
df.iloc[3:6]
df.iloc['2020-01-02': '2020-01-05'] # error!
df.iloc[0, 2] # get cell
df.iloc[1:4, 3:5] # get rows and cols interval
df.iloc[:, [1, 3]] # all rows, some cols


# import
df = pd.read_csv('https://raw.githubusercontent.com/rpalloni/dataset/master/titanic.csv')
# pd.set_option('display.max_columns', None) # full cols list

df.head()
df.loc[0] # first row
df.rename(columns={'pclass': 'class'}, inplace=True) # inplace sobstitute instead of create a copy
df.describe()

st = df.groupby(['class', 'sex']).agg({'age': 'mean', 'fare': 'mean'})
print(st)

pv = df.pivot_table(values=['age', 'fare'], index=['class', 'sex'], aggfunc='mean')
cr = pd.crosstab(df['class'], df['sex'], margins=True)
sr = df.sort_values(['fare'], ascending=False).head(10)

# export
ExcelObject = pd.ExcelWriter(path='stats.xlsx')
pv.to_excel(ExcelObject, sheet_name='pivoted', merge_cells=False) # repeat dimension level
cr.to_excel(ExcelObject, sheet_name='crossed')
sr.to_excel(ExcelObject, sheet_name='sorted')
ExcelObject.save()
