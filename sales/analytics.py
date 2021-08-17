import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

jan = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_January_2019.csv'
feb = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_February_2019.csv'
mar = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_March_2019.csv'
apr = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_April_2019.csv'
may = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_May_2019.csv'
jun = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_June_2019.csv'
jul = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_July_2019.csv'
aug = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_August_2019.csv'
sep = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_September_2019.csv'
oct = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_October_2019.csv'
nov = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_November_2019.csv'
dec = 'https://raw.githubusercontent.com/rpalloni/dataset/master/sales/Sales_December_2019.csv'

months = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]

sales = []
for m in months:
    mdf = pd.read_csv(m)
    mdf["Month"] = m[70:73] # get 3 chars month label
    sales.append(mdf)

df = pd.concat(sales)
df.head()

df.shape
df.columns
df.dtypes

# NAs
df.isnull().sum()
(df.isnull().sum().sum())/len(df)*100 # 1.75% null values => drop
df = df.dropna()

# check for anomalous data
df['Order ID'].unique()
df['Quantity Ordered'].unique()
df['Price Each'].unique()
df['Order Date'].unique()

df[df['Quantity Ordered'] == "Quantity Ordered"] # column header in data
df = df[df['Quantity Ordered'] != "Quantity Ordered"]

# numerical variables
df["Quantity Ordered"] = df["Quantity Ordered"].astype("float")
df["Price Each"] = df["Price Each"].astype("float")

# add sales column
df["Sales"] = df["Quantity Ordered"]*df["Price Each"]
sales_per_month = df.groupby(['Month']).agg({'Sales': 'sum'}).sort_values(by=['Sales'], ascending=False).reset_index()
sales_per_month

# color utils
color_map = plt.get_cmap("viridis")
def rescale(y): return (y - np.min(y)) / (np.max(y) - np.min(y))

# Sales per Month
fig, ax = plt.subplots(figsize=(16, 9))
ax.barh(
    sales_per_month['Month'],
    sales_per_month['Sales'],
    height=0.8, alpha=0.4,
    color=color_map(rescale(sales_per_month['Sales'])))
# remove borders
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)
ax.invert_yaxis()
ax.set_title('Total sales per month', loc='left', )
fig.text(0.9, 0.15, 'Source: db', fontsize=12,
         color='grey', ha='right', va='bottom',
         alpha=0.7)
ax.ticklabel_format(style='plain')
plt.show()


# Sales per City
