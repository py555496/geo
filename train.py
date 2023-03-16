#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 19:21:07 2021

@author: hkx
"""
import libpysal as ps
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import GWxgboost
from GWxgboost import GWxgboost as GWX
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False
import warnings 
#warnings.filterwarnings(action='ignore',module='LinAlgWarning')
import warnings
#import multiprocessing
warnings.filterwarnings("ignore")


#Data Preprocess
#data = pd.read_csv('data.csv')
data = pd.read_csv('sample_data.csv')
#data = pd.read_excel('sample_data.xlsx', sheet_name='Sheet1')

#去除id特征 市辖区 街道乡镇 因为是非数字特征，后期可以尝试转化成hot参与计算
#data = data.drop(columns=['id','distr','street'])
data = data.drop(columns=['distr','street', 'houset', 'price'])

#去除数据无效的空置
print(data.shape)
data = data.dropna()
c_name = data.columns[1:]
data.drop_duplicates(subset=c_name, keep = 'last', inplace=True)
print(data.shape)
ori_X = data.drop(columns=['rent'])
ori_Y = np.array(data.rent).reshape(-1,1).astype('float64')#

X_train, X_test, y_train, y_test = train_test_split(ori_X, ori_Y, test_size=0.2, random_state=1)

#获取geo地理信息
#coords = list(zip(data.x, data.y))#The coordinates of the observation
#data['x'] = (data['x'] - data['x'].min()) / (data['x'].max() - data['x'].min())
#data['y'] = (data['y'] - data['y'].min()) / (data['y'].max() - data['y'].min())

#coords = list(zip(data.x, data.y))#The coordinates of the observation
tr_coords = list(zip(X_train.x, X_train.y))#The coordinates of the observation
te_coords = list(zip(X_test.x, X_test.y))#The coordinates of the observation

Y = np.array(y_train).reshape(-1,1).astype('float64')#
#Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)


t_Y = np.array(y_test).reshape(-1,1).astype('float64')#
#t_Y = (t_Y - t_Y.mean(axis=0)) / t_Y.std(axis=0)

"""
data = data.drop(columns=['rent','x','y'])
X = np.array(data).reshape(-1,data.shape[1])
X = (X - X.mean(axis=0)) / X.std(axis=0)
"""
f_data = X_train.drop(columns=['x','y', 'id'])
X = np.array(f_data).reshape(-1, f_data.shape[1])
X = (X - X.mean(axis=0)) / X.std(axis=0)


test_f_data = X_test.drop(columns=['x','y', 'id'])
te_X = np.array(test_f_data).reshape(-1, test_f_data.shape[1])
te_X = (te_X - te_X.mean(axis=0)) / te_X.std(axis=0)

#normal xgboost
params = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10,
    'n_estimators': 10
}

# 训练模型
model = xgb.XGBRegressor(**params)
model.fit(f_data, Y)

# 预测测试集结果
y_pred = model.predict(test_f_data)

# 计算均方误差
mse = mean_squared_error(t_Y, y_pred)
print("base XGBOOST均方误差：", mse)

"""
gwr_selector = Sel_BW(tr_coords,Y,X)
gwr_bw = gwr_selector.search()
#算一次很久，所以保存当时值
print(gwr_bw)
"""
gwr_bw = 261.1



#Traditional GWR
#pool = multiprocessing.Pool()
model = GWR(tr_coords, Y, X, gwr_bw)
#results = model.fit(pool=pool)
results = model.fit()
#results.summary()

preds1=model.predict(np.array(te_coords), te_X)
mae = np.sqrt(mean_absolute_error(t_Y, preds1.predictions))
print("MAE: %f", mae)
rmse1 = np.sqrt(mean_squared_error(t_Y, preds1.predictions))
print("RMSE1: %f" % (rmse1))
r2 = r2_score(t_Y, preds1.predictions)
print("r2: %f", r2)

print("==" * 20)

feature_name_dict = {}
temp_arr = []
for line in open('./feature_name_dict', encoding='utf-8'):
    line = line.strip()
    temp_arr.append(line.split(","))
for i in range(len(temp_arr[0])):
    feature_name_dict[temp_arr[1][i]] = temp_arr[0][i]
print(feature_name_dict)



#GWxgboost
xgboster = GWX(data_x=X,data_y=Y,coords=tr_coords, bw=gwr_bw,special_objective='weighted',kernel="uniform")
xgboster.fit()
preds_2=xgboster.predict(te_X)
mae = np.sqrt(mean_absolute_error(t_Y, preds_2))
print("MAE: %f", mae)
rmse1 = np.sqrt(mean_squared_error(t_Y, preds_2))
print("RMSE1: %f" % (rmse1))
r2 = r2_score(t_Y, preds_2)
print("r2: %f", r2)
# plot the importance
importance = xgboster.regression_model.get_score(importance_type='weight')
#plt.bar(range(len(importance)),list(importance.values()),tick_label=list(importance.keys()))
feature_out = []
for g in importance.keys():
    index = int(g[1:])
    feature_out.append(f_data.columns[index])
feature_out = [feature_name_dict[x] for x in feature_out]
#plt.bar(range(len(importance)),list(importance.values()),tick_label=f_data.columns)
plt.bar(range(len(importance)),list(importance.values()),tick_label=feature_out)
plt.xticks(rotation=90)
plt.show()

#pd_results = partial_dependence(xgboster.regression_model, X, features=0, kind="average", grid_resolution=5)
"""



xgboster2 = GWX(data_x=X,data_y=Y,coords=coords, bw=gwr_bw)
xgboster2.fit()
preds2=xgboster2.predict(X)
rmse2 = np.sqrt(mean_squared_error(Y, preds2))
print("RMSE2: %f" % (rmse2))
importance2 = xgboster2.regression_model.get_score(importance_type='weight')
plt.bar(range(len(importance2)),list(importance2.values()),tick_label=list(importance2.keys()))
plt.xticks(rotation=90)
plt.show()
"""
