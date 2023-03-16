#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 19:21:07 2021

@author: hkx
"""
import libpysal as ps
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from GWxgboost import GWxgboost as GWX
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Data Preprocess
data = pd.read_csv('data.csv')
coords = list(zip(data.x, data.y))#The coordinates of the observation
Y = np.array(data.price).reshape(-1,1)#
data = data.drop(columns=['price','x','y'])
X = np.array(data).reshape(-1,data.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)


#Traditional GWR
gwr_selector = Sel_BW(coords,Y,X)
gwr_bw = gwr_selector.search()
model = GWR(coords, Y, X, gwr_bw)
results = model.fit()
results.summary()




#GWxgboost
xgboster = GWX(data_x=X,data_y=Y,coords=coords, bw=gwr_bw,special_objective='weighted',kernel="uniform")
xgboster.fit()
preds1=xgboster.predict(X)
rmse1 = np.sqrt(mean_squared_error(Y, preds1))
print("RMSE1: %f" % (rmse1))
# plot the importance
importance = xgboster.regression_model.get_score(importance_type='weight')
plt.bar(range(len(importance)),list(importance.values()),tick_label=list(importance.keys()))
plt.xticks(rotation=90)
plt.show()

pd_results = partial_dependence(xgboster.regression_model, X, features=0, kind="average", grid_resolution=5)




xgboster2 = GWX(data_x=X,data_y=Y,coords=coords, bw=gwr_bw)
xgboster2.fit()
preds2=xgboster2.predict(X)
rmse2 = np.sqrt(mean_squared_error(Y, preds2))
print("RMSE2: %f" % (rmse2))
importance2 = xgboster2.regression_model.get_score(importance_type='weight')
plt.bar(range(len(importance)),list(importance.values()),tick_label=list(importance.keys()))
plt.xticks(rotation=90)
plt.show()
