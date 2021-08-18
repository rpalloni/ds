import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

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

df[df['Quantity Ordered'] == 'Quantity Ordered'] # column header in data
df = df[df['Quantity Ordered'] != 'Quantity Ordered']

# numerical variables
df['Quantity Ordered'] = df['Quantity Ordered'].astype('float')
df['Price Each'] = df['Price Each'].astype('float')

# Sales per Month
df['Sales'] = df['Quantity Ordered']*df['Price Each']
sales_per_month = df.groupby(['Month']).agg({'Sales': 'sum'}).sort_values(by=['Sales'], ascending=False).reset_index()
sales_per_month['% of Total'] = sales_per_month['Sales'] / sales_per_month['Sales'].sum()
sales_per_month.style.format({'Sales': '{:,.2f}', '% of Total': '{:.2%}'})

# color utils
color_map = plt.get_cmap('cividis')
def rescale(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y))

# plot
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
ax.set_title('Total sales per month', loc='center', )
ax.set_xlabel('Amount')
ax.set_ylabel('Months')
fig.text(0.9, 0.15, 'Source: db', fontsize=12,
         color='grey', ha='right', va='bottom',
         alpha=0.7)
ax.ticklabel_format(axis='x', style='plain')
plt.show()


# Sales per City
df['City'] = df['Purchase Address'].apply(lambda x: x.split(',')[1])
sales_per_city = df.groupby(['City']).agg({'Sales': 'sum'}).sort_values(by=['Sales'], ascending=False).reset_index()
sales_per_city.style.format({'Sales': '{:,.2f}'}).highlight_max(color='green').highlight_min(color='red')

sales_per_city.style.format({'Sales': '{:,.2f}'}).background_gradient(subset=['Sales'], cmap='GnBu')

# plot
fig, ax = plt.subplots(figsize=(16, 9))
ax.bar(
    sales_per_city['City'],
    sales_per_city['Sales'],
    alpha=0.4, color=color_map(rescale(sales_per_month['Sales'])))
# remove borders
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)
ax.set_title('Total sales per city', loc='center', )
ax.set_xlabel('City')
ax.set_ylabel('Amount')
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: '{:.0f}'.format(value/1000000) + ' M'))
fig.text(0.9, 0.9, 'Source: db', fontsize=12,
         color='grey', ha='right', va='top',
         alpha=0.7)
#ax.ticklabel_format(axis='y', style='plain')
plt.show()


# Best product
best_product = df.groupby(['Product']).agg({'Quantity Ordered': 'sum'}).sort_values(by=['Quantity Ordered'], ascending=False).reset_index()
best_product

# Best sales timing
df['Sale Hour'] = df['Order Date'].apply(lambda x: x.split(' ')[1]).str[0:2]
best_sales_hr = df.groupby(['Sale Hour']).agg({'Sales': 'sum'}).sort_values(by=['Sales'], ascending=False).reset_index()
best_sales_hr.head()

# plot
qnt_hr = df.groupby(['Sale Hour']).agg({'Quantity Ordered': 'count'})
hr = [hr for hr, df in df.groupby('Sale Hour')]
plt.plot(hr, qnt_hr)
plt.xlabel('Hour')
plt.ylabel('Quantity')
plt.grid()


### Product basket
# number of products per order
df.groupby('Order ID').agg({'Product': 'count'}).sort_values(by=['Product'], ascending=False).reset_index()
# order with max number of products
df[df['Order ID'] == '160873']
# extract only multiproduct orders
multiproduct_orders = df[df['Order ID'].duplicated(keep=False)]
# combine products to create a new grouping column
multiproduct_orders['Product Basket'] = multiproduct_orders.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# drop order duplicates
multiproduct_orders = multiproduct_orders.drop_duplicates(subset=['Order ID', 'Product Basket'])
multiproduct_orders[multiproduct_orders['Order ID'] == '160873']

multiproduct_orders.groupby(['Product Basket'])['Order ID'].count().sort_values(ascending=False).reset_index()

# caveats
multiproduct_orders[multiproduct_orders['Product Basket'] == 'iPhone,Lightning Charging Cable']['Order ID'].count()
multiproduct_orders[multiproduct_orders['Product Basket'] == 'Lightning Charging Cable,iPhone']['Order ID'].count()

multiproduct_orders[multiproduct_orders['Product Basket'].str.contains('iPhone,Lightning Charging Cable')]['Order ID'].count()
multiproduct_orders[multiproduct_orders['Product Basket'].str.contains('Lightning Charging Cable,iPhone')]['Order ID'].count()
