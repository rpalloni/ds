# data source
# https://cohesiondata.ec.europa.eu/dataset/ESIF-2014-2020-FINANCES-PLANNED-DETAILS/e4v6-qrrq
# https://cohesiondata.ec.europa.eu/EU-Level/ESIF-2014-2020-Finance-Implementation-Details/99js-gm52

import requests as query
import pandas as pd

pd.set_option('display.float_format', lambda x: '%10.2f' % x) # '10.2f' float two decimals
pd.set_option('display.max_columns', None) # display all columns

# json endpoints Planned and Implemented
# PlanUrl = 'https://cohesiondata.ec.europa.eu/resource/rde7-u3r9.json'
ImplUrl = 'https://cohesiondata.ec.europa.eu/resource/f6wa-fhmb.json?$limit=1000000'
JsonData = query.get(ImplUrl).json()
df = pd.DataFrame(JsonData)
# df = pd.read_csv('ESIF_2014-2020_Finance_Implementation_Details.csv', delimiter=',', thousands=',')
df.shape
print(df.dtypes)

# import variable names and types
TypeUrl = 'https://raw.githubusercontent.com/rpalloni/dataset/master/TypeSource.csv'
TypeString = pd.read_csv(TypeUrl, delimiter=';')
print(TypeString)

VarType = dict(zip(list(TypeString.Variable), list(TypeString.Type)))

dt = df.astype(dtype=VarType) # pd.DataFrame(JsonData, dtype={...}) not supported
print(dt.dtypes)
dt.head(20)

tot = dt.groupby(['ms_name', 'fund', 'year']).agg({'total_eligible_cost': 'sum', 'total_eligible_expenditure': 'sum'})
pv = dt.pivot_table(values=['total_eligible_cost', 'total_eligible_expenditure'], index=['ms_name', 'fund', 'year'], aggfunc='sum')
print(tot)
print(pv)

# XlsxWriter https://pypi.org/project/XlsxWriter/
ExcelObject = pd.ExcelWriter(path='EsifTabs.xlsx', engine='xlsxwriter')
tot.to_excel(ExcelObject, sheet_name='totaled', merge_cells=False)
pv.to_excel(ExcelObject, sheet_name='pivoted', merge_cells=True)
ExcelObject.save()


# last year operations
MStot2017 = dt.loc[(dt['year'] > 2016) & (dt['ms_name'] != 'Territorial co-operation')].groupby(['ms_name']).agg({'total_eligible_cost': 'sum', 'total_eligible_expenditure': 'sum'}) # () for multiple conds
MSpv2017 = dt[(dt['year'] > 2016) & (dt['ms_name'] != 'Territorial co-operation')].pivot_table(values=['total_eligible_cost', 'total_eligible_expenditure'], index=['ms_name'], aggfunc='sum')

ExcelObject = pd.ExcelWriter(path='EsifFigure.xlsx', engine='xlsxwriter')
MStot2017.to_excel(ExcelObject, sheet_name='totaled')
MSpv2017.to_excel(ExcelObject, sheet_name='pivoted')

### Add a chart based on data
# Select workbook and worksheet objects
wb = ExcelObject.book
ws = ExcelObject.sheets['pivoted']


cell_format = wb.add_format({'bold': True, 'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
ws.conditional_format('B2:B29', {'type': 'cell',
                                 'criteria': '>=',
                                 'value': 12000000000,
                                 'format': cell_format})

# Create a chart object
chart = wb.add_chart({'type': 'column'}) # bar, supported types: https://xlsxwriter.readthedocs.io/chart.html#chart-class
# Configure the series of the chart from the dataframe data
chart.add_series({'name': 'Project Selection', 'categories': '=pivoted!A2:A29', 'values': '=pivoted!B2:B29', 'fill': {'color': 'green'}})
chart.add_series({'name': 'Project Expenditure', 'categories': '=pivoted!A2:A29', 'values': '=pivoted!C2:C29', 'fill': {'color': 'red'}})
# Add a chart title and some axis labels
chart.set_title({'name': 'Project Selection and Expenditure'})
chart.set_x_axis({'name': 'MS'})
chart.set_y_axis({'name': 'EUR'})
chart.set_legend({'position': 'bottom'})
# Insert the chart into the worksheet cell
ws.insert_chart('F2', chart)

ExcelObject.save()
