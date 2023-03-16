import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.metrics import mean_squared_error, r2_score

# 创建模拟数据集
np.random.seed(42)
n = 100
lat = np.random.uniform(0, 100, n)
lon = np.random.uniform(0, 100, n)
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(10, 20, n)
y = 5 + 2 * x1 + 3 * x2 + 0.5 * lat + 0.5 * lon + np.random.normal(0, 1, n)

# 创建DataFrame
data = pd.DataFrame({'lat': lat, 'lon': lon, 'x1': x1, 'x2': x2, 'y': y})
coords = data[['lat', 'lon']].values
X = data[['x1', 'x2']].values
y = data['y'].values

# 将y转换为二维NumPy数组
y = y.reshape(-1, 1)

# 选择GWR模型的最佳带宽
sel_bw = Sel_BW(coords, y, X)
bw = sel_bw.search()

# 使用最佳带宽拟合GWR模型
gwr_model = GWR(coords, y, X, bw)
gwr_results = gwr_model.fit()

# 获取预测值
y_pred = gwr_model.predict(coords, X)
#y_pred = gwr_results.predicted()

# 评估
mse = mean_squared_error(y, y_pred.predictions)
r2 = r2_score(y, y_pred.predictions)

print('MSE:', mse)
print('R2:', r2)
