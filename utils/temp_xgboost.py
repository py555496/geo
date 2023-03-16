import xgboost as xgb
import matplotlib.pyplot as plt

# 加载数据
data = xgb.DMatrix('data.csv')

# 定义参数
params = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic'}

# 训练模型
model = xgb.train(params, data)

# 获取特征重要性得分
scores = model.get_score()

# 将得分可视化
plt.bar(range(len(scores)), list(scores.values()), tick_label=list(scores.keys()))
plt.show()
