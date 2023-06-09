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
#data = pd.read_csv('all_data.csv')
data = pd.read_csv('sample_data.csv')

#去除id特征 市辖区 街道乡镇 因为是非数字特征，后期可以尝试转化成hot参与计算
useness_feature = ['distr','street', 'houset', 'price']
#data = data.drop(columns=['id','distr','street'])
data = data.drop(columns=useness_feature)

#去除数据无效的空置
print(data.shape)
data = data.dropna()
c_name = data.columns[1:]
data.drop_duplicates(subset=c_name, keep = 'last', inplace=True)
print(data.shape)
ori_X = data.drop(columns=['rent'])
ori_Y = np.array(data.rent).reshape(-1,1).astype('float64')#

X_train, X_test, y_train, y_test = train_test_split(ori_X, ori_Y, test_size=0.15, random_state=1)

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
#un_geo_feature = ['x','y', 'id', 'medicad', 'area', 'volume', 'year']
un_geo_feature = ['x','y', 'id', 'medicad']
f_data = X_train.drop(columns=un_geo_feature)
X = np.array(f_data).reshape(-1, f_data.shape[1])
#debug 不在特征空间上归一化
X = (X - X.mean(axis=0)) / X.std(axis=0)


test_f_data = X_test.drop(columns=un_geo_feature)
te_X = np.array(test_f_data).reshape(-1, test_f_data.shape[1])
te_X = (te_X - te_X.mean(axis=0)) / te_X.std(axis=0)

#normal xgboost
params = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.2,
    'max_depth': 3,
    'alpha': 10,
    'n_estimators': 10
}

N = 50

# 训练模型
model = xgb.XGBRegressor(**params)
model.fit(f_data, Y)

# 预测测试集结果
base_y_pred = model.predict(test_f_data)

# 计算均方误差
mae = np.sqrt(mean_absolute_error(t_Y, base_y_pred))
mse = np.sqrt(mean_squared_error(t_Y, base_y_pred))
#r2 = r2_score(t_Y, base_y_pred)
import numpy as np

print(t_Y.shape)
def r_squared(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    residual = np.subtract(y_true, y_pred)
    total = np.subtract(y_true, np.mean(y_true))
    r2 = 1 - (np.sum(np.square(residual)) + 0.001) / (np.sum(np.square(total)) + 0.001 )
    return r2
r2 = r_squared(t_Y[:N], base_y_pred[:N])

temp_x = [x for x in range(N)]
print("base XGBOOST MAE: {:.5f}".format(mae))
print("base XGBOOST RMSE：{:.5f}".format(mse))
#print("base XGBOOST R^2：%f", r2)
print("base XGBOOST R^2：{:.5f}".format(r2))
print("==" * 40)

"""
gwr_selector = Sel_BW(tr_coords,Y,X)
gwr_bw = gwr_selector.search()
#算一次很久，所以保存当时值
print(gwr_bw)
"""
gwr_bw = 6700.1



#Traditional GWR
#多线程有问题
#multiprocessing.freeze_support()
#pool = multiprocessing.Pool(processes=3)
model = GWR(tr_coords, Y, X, gwr_bw)
#results = model.fit(pool=pool)
results = model.fit()
#results.summary()

preds1=model.predict(np.array(te_coords), te_X)
mae = np.sqrt(mean_absolute_error(t_Y, preds1.predictions))
print("GWR MAE: {:.5f}".format(mae))

rmse1 = np.sqrt(mean_squared_error(t_Y, preds1.predictions))
print("GWR RMSE: {:.5f}".format(rmse1))
r2 = r_squared(t_Y, preds1.predictions)
print("GWR R^2: {:.5f}".format(r2))
#print(t_Y[:N])
#print(preds1.predictions[:N])

print("==" * 40)

#方便显示 特征标签
feature_name_dict = {}
temp_arr = []
for line in open('./feature_name_dict', encoding='utf-8'):
    line = line.strip()
    temp_arr.append(line.split(",")) 
for i in range(len(temp_arr[0])):
    feature_name_dict[temp_arr[1][i]] = temp_arr[0][i]


#GWxgboost
xgboster = GWX(data_x=X,data_y=Y,coords=tr_coords, bw=gwr_bw, num_round=100, special_objective='weighted',kernel="uniform")
xgboster.fit()
preds_2=xgboster.predict(te_X)
mae = np.sqrt(mean_absolute_error(t_Y, preds_2))
print("GWX MAE: {:.5f}".format(mae))
rmse1 = np.sqrt(mean_squared_error(t_Y, preds_2))
print("GWX RMSE: {:.5f}".format(rmse1))
r2 = r_squared(t_Y, preds_2)
print("GWX R^2: {:.5f}".format(r2))
# plot the importance
importance = xgboster.regression_model.get_score(importance_type='weight')
#plt.bar(range(len(importance)),list(importance.values()),tick_label=list(importance.keys()))
feature_out = []
for g in importance.keys():
    index = int(g[1:])
    feature_out.append(f_data.columns[index])

"""
plt.plot(temp_x, t_Y[:N], color='black')
plt.plot(temp_x, preds1.predictions[:N], color='blue')
plt.plot(temp_x, base_y_pred[:N], color='red')
plt.plot(temp_x, preds_2[:N], color='green')
"""



feature_out = [feature_name_dict[x] for x in feature_out]

im_ratio = list(importance.values())
sum_im = sum(im_ratio)
im_ratio = [x / sum_im for x in im_ratio]
im_ratio_value = ["{:.2f}".format(x / sum_im) for x in im_ratio]
plt.bar(range(len(importance)), im_ratio, tick_label=feature_out)
#for i in range(len(importance)):
#    plt.text(range(len(importance))[i], im_ratio_value[i], im_ratio_value[i], ha='center', va='bottom')
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
