import pandas as pd  # 读取 xlsx 文件 
df = pd.read_excel('zong_data.xlsx', sheet_name='3')
# 将数据保存为 csv 文件 
df.to_csv('zong_data.csv', index=False)
