import pandas as pd  # 读取 xlsx 文件 
df = pd.read_excel('all_data.xlsx', sheet_name='Sheet7')
# 将数据保存为 csv 文件 
df.to_csv('new_all_data.csv', index=False)
