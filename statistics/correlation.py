
import io
import math
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as p
import requests

download_url = "https://raw.githubusercontent.com/rpalloni/dataset/master/airquality.csv"
response = requests.get(download_url)
data = pd.read_csv(io.BytesIO(response.content), dtype={'Ozone': float,'SolarRay':float, 'Wind':float, 'Temp':float, 'Month':float, 'Day':float})
data.head()

data.isnull().values.any()
data = data.dropna()

ozone = data['Ozone'].values
solar = data['SolarRay'].values
wind = data['Wind'].values

### covariance explained
# [2x2]
# varx  covxy
# covxy vary
n = len(ozone)
mean_x, mean_y = sum(ozone) / n, sum(solar) / n
cov_xy = (sum((ozone[k] - mean_x) * (solar[k] - mean_y) for k in range(n)) / (n - 1))
cov_xy

# r^2
var_x = sum((item - mean_x)**2 for item in ozone) / (n - 1)
var_y = sum((item - mean_y)**2 for item in ozone) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r


### numpy
ozone.var(ddof=1)
solar.var(ddof=1)

cov_xy = np.cov(ozone,solar)

cov_xy[0,1] == cov_xy[1,0]

np.corrcoef(ozone,solar) # pearson
xyz = (ozone,solar,wind)
np.corrcoef(xyz) # correlation matrix

### scipy
scipy.stats.pearsonr(ozone, solar)    # Pearson's r
scipy.stats.spearmanr(ozone, solar)   # Spearman's rho
scipy.stats.kendalltau(ozone, solar)  # Kendall's tau

scipy.stats.pearsonr(ozone, solar)[0]
scipy.stats.spearmanr(ozone, solar).correlation

# tuple of results
r, pv = scipy.stats.pearsonr(ozone, solar)
r
pv

### pandas
data.corr() # full correlation matrix

ozone = data['Ozone']
solar = data['SolarRay']
wind = data['Wind']

ozone.corr(solar, method='pearson')
ozone.corr(solar, method='spearman')
ozone.corr(solar, method='kendall')


# linear regression
result = scipy.stats.linregress(data['SolarRay'], data['Ozone'])

result.slope
result.intercept
result.rvalue
result.pvalue
result.stderr

# scatterplot
line = f'Regression line: y={result.intercept:.2f}+{result.slope:.2f}x, r={result.rvalue:.2f}'

# heatmap
matrix = np.cov(ozone, solar).round(decimals=2)
fig, ax = p.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('ozone', 'solar'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('ozone', 'solar'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
p.show()



matrix = np.cov(data.corr()).round(decimals=2)
fig, ax = p.subplots()
im = ax.imshow(matrix)
#im.set_clim(-1, 1)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2, 3, 4, 5), ticklabels=('ozone', 'solar', 'wind', 'temp', 'month', 'day'))
ax.yaxis.set(ticks=(0, 1, 2, 3, 4, 5), ticklabels=('ozone', 'solar', 'wind', 'temp', 'month', 'day'))
ax.set_ylim(5.5, -0.5)
for i in range(6):
    for j in range(6):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
p.show()




#rank-correlation