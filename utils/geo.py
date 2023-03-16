import matplotlib.pyplot as plt
import pandas as pd
# 加载数据集
#Data Preprocess
#data = pd.read_csv('data.csv')
data = pd.read_excel('new_data.xlsx', sheet_name='Sheet6')
#data = pd.read_excel('sample_data.xlsx', sheet_name='Sheet1')

#去除id特征 市辖区 街道乡镇 因为是非数字特征，后期可以尝试转化成hot参与计算
#data = data.drop(columns=['id','dist','str'])
x = data.x
y = data.y
# 坐标数据

# 绘制散点图
plt.scatter(x, y)

# 设置图表标题和坐标轴标签
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示图表
plt.show()
