import io
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import requests

download_url = "https://raw.githubusercontent.com/rpalloni/dataset/master/airquality.csv"
response = requests.get(download_url)
data = pd.read_csv(io.BytesIO(response.content), dtype={'Ozone': float,'SolarRay':float, 'Wind':float, 'Temp':float, 'Month':float, 'Day':float})
data.head()

data.isnull().values.any()
data = data.dropna()


ozone = data['Ozone']
ozone = data.Ozone
ozone = data.filter(like='ay')

ozone = data['Ozone'].values


### measures-of-central-tendency

# mean
sum(ozone)/len(ozone)
statistics.mean(ozone)
np.mean(ozone)
np.nanmean(ozone)
ozone.mean() # pandas


# harmonic
len(ozone) / sum(1/item for item in ozone)
statistics.harmonic_mean(ozone)
scipy.stats.hmean(ozone)

# geometric
gmean = 1
for item in ozone: gmean *= item
gmean **= 1 / len(ozone)
gmean
scipy.stats.gmean(ozone)

# median
n = len(ozone)
if n % 2:
    # n even
    med = sorted(ozone)[round(0.5*(n-1))]
else:
    # n odd
    ord = sorted(ozone)
    idx = round(0.5*n)
    med = 0.5 *(ord[idx-1] + ord[idx])
med

statistics.median(ozone)
statistics.median(ozone[:-1]) # exclude last item
np.median(ozone)
data['Ozone'].median()

# mode
statistics.mode(ozone)
data['Ozone'].mode()


### measures-of-variability

# variance
n = len(ozone)
m = sum(ozone)/n
sum((item-m)**2 for item in ozone) / (n-1)

statistics.variance(ozone)
np.var(ozone, ddof=1) # degrees of freedom: use (n-1) instead of n
statistics.pvariance(ozone) # population variance (ddof=0)
np.nanvar(ozone, ddof=1)
ozone.var(ddof=1)

# standard deviation
math.sqrt(statistics.variance(ozone))
statistics.variance(ozone)**0.5

statistics.stdev(ozone)
statistics.pstdev(ozone) # population sd
np.std(ozone, ddof=1)
ozone.std(ddof=1)

# skewness
n = len(ozone)
mean = sum(ozone)/n
var = sum((item-mean)**2 for item in ozone) / (n-1)
std = var ** 0.5
skw = (sum((item - mean)**3 for item in ozone) * n / ((n - 1) * (n - 2) * std**3))
skw # positive: rigth-side skw

scipy.stats.skew(ozone, bias=False)
data['Ozone'].skew()

# percentiles
statistics.quantiles(ozone, n=2)
statistics.quantiles(ozone, n=4, method='inclusive')

np.quantile(ozone, 0.05)
np.quantile(ozone, 0.95)

np.percentile(ozone, [25,50,75])
np.nanpercentile(ozone, [25,50,75])

ozone.quantile(0.05)
ozone.quantile(0.95)

# range min-max
ozone.max() - ozone.min()

quartiles = np.quantile(ozone, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = ozone.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]


### summary-of-descriptive-statistics

# describe
dstats = scipy.stats.describe(ozone, ddof=1, bias=False)
dstats.nobs
dstats.minmax[0]
dstats.minmax[1]
dstats.mean
dstats.variance
dstats.skewness
dstats.kurtosis

data['Ozone'].describe() # pandas

from collections import namedtuple

def describe(sample):
    Desc = namedtuple("Desc", ["mean", "median", "mode"])
    return Desc(
        statistics.mean(sample),
        statistics.median(sample),
        statistics.mode(sample),
    )

describe(data['Ozone'].dropna())


### measures-of-correlation-between-pairs-of-data

# covariance

ozone = data['Ozone'].values
solar = data['SolarRay'].values

len(ozone)
len(solar)

# covariance [2x2]
# varx  covxy
# covxy vary
n = len(ozone)
mean_x, mean_y = sum(ozone) / n, sum(solar) / n
cov_xy = (sum((ozone[k] - mean_x) * (solar[k] - mean_y) for k in range(n)) / (n - 1))
cov_xy

ozone.var(ddof=1)
solar.var(ddof=1)

cov_xy = np.cov(ozone,solar)

cov_xy[0,1] == cov_xy[1,0]

# r^2
var_x = sum((item - mean_x)**2 for item in ozone) / (n - 1)
var_y = sum((item - mean_y)**2 for item in ozone) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

scipy.stats.pearsonr(ozone, solar) # r, p-value

scipy.stats.linregress(ozone, solar)

ozone = data['Ozone']
solar = data['SolarRay']
ozone.corr(solar)

# dataframes 

data.mean() # cols mean
data.mean(axis=1) # rows mean

data['Ozone'].mean()

data.describe()

data.describe().at['mean', 'Ozone']
