#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
data = pd.read_csv('zong_data.csv')
"""
#result = data.groupby('distr').agg({'street': 'nunique'})
result = data.groupby('distr').size()
#result = result.rename(columns={'street': 'num_subregions'})
##样本数
#print(result)
result = data.groupby('distr').agg({'street': 'nunique'})
result = result.rename(columns={'subregion': 'num_subregions'})
#每个取的街道数
print('每个大区的街道数')
print(result)
print("++" * 50)
total_records = len(data)
result = data.groupby('distr').size()
result = result.rename('num_records').reset_index()
result['percent'] = result['num_records'] / total_records * 100
print('每个大区的样本数据和所占比例')
print(result)
print("++" * 50)
result = data.groupby('distr')['rent'].mean()
print('每个大区的平均租金')
print(result)
"""
print("每个大区的设备数量")
para_arr = ['distr', 'street', 'gym', 'square','bus', 'sub', 'hosp', 'kinder', 'prim', 'high', 'mall', 'super'] 
df = pd.read_csv('zong_data.csv', usecols=para_arr) 
d= {}
for g in para_arr:
    if g != 'distr' and g != 'street':
        d[g] = 'sum'
df.drop_duplicates(inplace=True) 
#out = df.groupby('distr').agg({'gym': 'sum', 'kinded': 'sum'})
out = df.groupby('distr').agg(d)
out.to_excel('output.xlsx', index=True)
print(out)
