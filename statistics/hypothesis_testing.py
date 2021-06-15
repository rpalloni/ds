import pandas as pd
import scipy.stats as sts
from scipy.stats import ttest_1samp, ttest_ind
import math

'''
Null hypotheses: population mean = 7
Alternative hypotheses: population mean > 7

Assumption 1: indipendent observations (random sample)
Assumption 2: sample size sufficiently large (>30)
Assumption 3: data normally distributed
'''
sleep_hrs = pd.read_csv("https://raw.githubusercontent.com/rpalloni/dataset/master/sleep_hrs.csv")

sleep_hrs.hist(figsize=(15,10), color='red', bins=9)

h0 = 7
n = sleep_hrs.count()
mu = sleep_hrs.mean().round(2)
sdv = sleep_hrs.std().round(2)
se = (sdv/math.sqrt(n)).round(2)
zscore = (mu-h0)/se # t_value: sample mean is 2.47 standard error above the null h0
p = sts.norm.sf(abs(zscore)) # one-sided p_value

if p < 0.05:    # alpha value is 0.05 or 5%
    print(f"p-value:{p.round(3)} - rejecting null hypothesis")
else:
    print(f"p-value:{p.round(3)} - accepting null hypothesis")

sts.ttest_1samp(sleep_hrs,7)

# If the null hypotesis is true (population mean = 7), the distribution of sample means from this population
# would spread around 7. since the probability of observing  such a large sample mean (7.42) is only 0.007
# we reject the null hypothesis


'''
Null hypotheses: Two group means are equal
Alternative hypotheses: Two group means are different (two-tailed)

Assumption 1: Are the two samples independent?
Assumption 2: Are the data from each of the 2 groups following a normal distribution?
Assumption 3: Do the two samples have the same variances (Homogeneity of Variance)?
'''
df = pd.read_csv("https://raw.githubusercontent.com/rpalloni/dataset/master/t_test.csv")

male = df.query('grouping == "men"')['height']
female = df.query('grouping == "women"')['height']

df.groupby('grouping').describe()

# check data normally distributed
# Shapiro-Wilks test: null hypothesis is normal distribution
sts.shapiro(male) # (t-value, p-value)
sts.shapiro(female)

# check homogeneity of variance
# Lavene test: null hypothesis is equal variance
sts.levene(male, female)

# two-sample t-test with scipy
res = sts.ttest_ind(male, female, equal_var=True)
display(res)
# p-value < 0.05: men’s average height is statistically different from the female’s average height


''' Type 2 Error:  do not reject the null when it is false '''
import numpy as np
import matplotlib.pyplot as plt
import scipy

# (unknown) population
mu, sigma = 3, 2
s = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()

# sample
sample_mean, sample_sigma = 1.5, 2
sample = np.random.normal(sample_mean, sample_sigma, 200)

count, bins, ignored = plt.hist(s, 30, alpha=0.1, density=True)
sample_count, sample_bins, sample_ignored = plt.hist(sample, 30, alpha=0.1, color='r',density=True)
plt.plot(sample_bins,1/(sample_sigma * np.sqrt(2 * np.pi)) *np.exp( - (sample_bins - sample_mean)**2 / (2 * sample_sigma**2) ),linewidth=2, color='r')
plt.plot(bins,1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b')
plt.show() # red sample, blue pop

'''
Null hypotheses: sample is taken from population with mean = 1.5
Alternative hypotheses: sample is taken from population with mean <> 1.5
'''

# confidence interval
ci = scipy.stats.norm.interval(0.95, loc=1.5, scale=2)
count, bins, ignored = plt.hist(s, 30, alpha=0.1, density=True)
sample_count, sample_bins, sample_ignored = plt.hist(sample, 30, alpha=0.1, color='r',density=True)
plt.plot(sample_bins,1/(sample_sigma * np.sqrt(2 * np.pi)) *np.exp( - (sample_bins - sample_mean)**2 / (2 * sample_sigma**2) ),linewidth=2, color='r')
plt.plot(bins,1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b')
plt.axvline(ci[0],color='g')
plt.axvline(ci[1],color='g')
plt.show()

# type I and type II errors
count, bins, ignored = plt.hist(s, 30, alpha=0.1, density=True)
sample_count, sample_bins, sample_ignored = plt.hist(sample, 30, alpha=0.1, color='r',density=True)
plt.plot(sample_bins,1/(sample_sigma * np.sqrt(2 * np.pi)) *np.exp( - (sample_bins - sample_mean)**2 / (2 * sample_sigma**2) ),linewidth=2, color='r')
plt.plot(bins,1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b')
plt.axvline(ci[0],color='g')
plt.axvline(ci[1],color='g')
plt.fill_between(x=np.arange(-4,ci[0],0.01),
                 y1= scipy.stats.norm.pdf(np.arange(-4,ci[0],0.01),loc=1.5,scale=2) ,
                 facecolor='red',
                 alpha=0.35)

plt.fill_between(x=np.arange(ci[1],7.5,0.01),
                 y1= scipy.stats.norm.pdf(np.arange(ci[1],7.5,0.01),loc=1.5,scale=2) ,
                 facecolor='red',
                 alpha=0.5)

plt.fill_between(x=np.arange(ci[0],ci[1],0.01),
                 y1= scipy.stats.norm.pdf(np.arange(ci[0],ci[1],0.01),loc=3, scale=2) ,
                 facecolor='blue',
                 alpha=0.5)

plt.text(x=0, y=0.18, s= "Null Hypothesis")
plt.text(x=6, y=0.05, s= "Alternative")
plt.text(x=-4, y=0.01, s= "Type 1 Error")
plt.text(x=6.2, y=0.01, s= "Type 1 Error")
plt.text(x=2, y=0.02, s= "Type 2 Error")

plt.show()

z_score=(sample_mean-mu)/sigma
p_value = scipy.stats.norm.sf(abs(z_score))
z_score
p_value # do not reject H0 but it is false (population mean is 3)
