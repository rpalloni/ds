import pandas as pd
import datetime as dt

df = pd.read_csv('sales.csv')
df.shape

# replacing the spaces in the column names to underscore
df.columns = df.columns.str.replace(' ', '_')

#convert order date to datetime
df['Order_Date'] = df['Order_Date'].astype('datetime64[ns]')

#convert ship date to datetime
df['Ship_Date'] = df['Ship_Date'].astype('datetime64[ns]')

#drop row id
df.drop('Row_ID', axis=1, inplace=True)

#sales per category and subcategory
s_cat = df.groupby(['Category', 'Sub_Category']).agg({'Sales': sum})

#sales per state and city
s_geo = df.groupby(['State']).agg({'Sales': sum}).sort_values(by=['Sales'], ascending=True)

#create report
ExcelObject = pd.ExcelWriter(path='SalesReport.xlsx')
s_cat.to_excel(ExcelObject, sheet_name='categories', merge_cells=True)
s_geo.to_excel(ExcelObject, sheet_name='geography')

wb = ExcelObject.book
values_format = wb.add_format({'num_format': '$#,##0.00'})

wsc = ExcelObject.sheets['categories']
chart_cat = wb.add_chart({'type': 'column'})
wsg = ExcelObject.sheets['geography']
chart_geo = wb.add_chart({'type': 'bar'})

# category
wsc.set_column('A:B', 25)
wsc.set_column('C:C', 20, values_format)

cond_format = wb.add_format({'bold': True, 'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
wsc.conditional_format('C2:C18', {'type': 'cell',
                                  'criteria': '>=',
                                  'value': 200000,
                                  'format': cond_format})


chart_cat.set_title({'name': 'Sales by Category and Sub Category'})
chart_cat.set_legend({'position': 'none'})
chart_cat.add_series({
    'categories': '=categories!A2:B18',
    'values':     '=categories!C2:C18',
})

wsc.insert_chart('F2', chart_cat)

# geography
wsg.set_column('A:A', 25)
wsg.set_column('B:B', 20, values_format)

chart_geo.set_size({'width': 320, 'height': 960})
chart_geo.set_title({'name': 'Sales by State'})
chart_geo.set_legend({'position': 'none'})
chart_geo.add_series({
    'categories': '=geography!A2:A50',
    'values':     '=geography!B2:B50',
})

wsg.insert_chart('F2', chart_geo)

ExcelObject.save()
