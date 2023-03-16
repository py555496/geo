import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成随机数据
# 加载数据集
#Data Preprocess
#data = pd.read_csv('data.csv')
data = pd.read_excel('new_data.xlsx', sheet_name='Sheet6')
#data = pd.read_excel('sample_data.xlsx', sheet_name='Sheet1')

#去除id特征 市辖区 街道乡镇 因为是非数字特征，后期可以尝试转化成hot参与计算
#data = data.drop(columns=['id','dist','str'])
x = data.x
y = data.y

# 绘制热力图
plt.imshow(zip(x,y), cmap='hot', interpolation='nearest')

# 显示颜色条
plt.colorbar()

# 显示图像
plt.show()
