
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:16:21 2021

@author: hkx
"""

#The model is based on the Xgboost package(https://xgboost.readthedocs.io/en/latest/python/index.html) and Sklearn package(https://scikit-learn.org/)
import numpy as np
import xgboost as xgb
from weighted_loss import Geographicaly_Weighted_Loss
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from kernels import *

#class GWxgboost (BaseEstimator, RegressorMixin):
class GWxgboost (RegressorMixin):
    """Data in the form of [nData * nDim], where nDim stands for the number of features.
       This wrapper would provide a Xgboost interface with sklearn estimiator structure, which could be stacked in other Sk pipelines
    """
    def __init__(self, data_x, data_y,coords,bw,num_round=15, max_depth=25, eta=0.3, verbosity=1, objective_func='reg:squarederror',kernel=None,
                 eval_metric='logloss', booster='gbtree', special_objective=None, early_stopping_rounds=None,points=None
                 ):
        """
         Parameters to initialize a Xgboost estimator
        :param data_x. The Training data independent variable
        :param data_y. The Training data dependent varibale
        :param coords. The coordinates of the data point which is corresponding to the training data
        :param bw. The bandwidth
        :param num_round. The rounds we would like to iterate to train the model
        :param max_depth. The maximum depth of the classification boosting, need to be specified
        :param eta Step. Size shrinkage used in update to prevents overfitting
        :param verbosity. Set to '1' or '0' to determine if print the information during training. True is higly recommended
        :param objective_func. The objective function we would like to optimize
        :param kernel type of kernel function used to weight observations
              available options: 'gaussian' 'bisquare' 'exponential'
        :param eval_metric. The loss metrix. Note this is partially correlated to the objective function, and unfit loss function would lead to problematic loss
        :param booster. The booster to be usde, can be 'gbtree', 'gblinear' or 'dart'.
        :param imbalance_alpha. The \alpha value for imbalanced loss. Will make impact on '1' classes. Must have when special_objective 'weighted'
        :param Kernel. The kernel function to be used
        """
        self.data_x=data_x
        self.data_y=data_y
        self.coords=coords
        self.points=points
        self.bw=bw
        self.num_round = num_round
        self.max_depth = max_depth
        self.eta = eta
        self.verbosity = verbosity
        self.objective_func = objective_func
        self.kernel=kernel
        self.eval_metric = eval_metric
        self.booster = booster
        self.eval_list = []
        self.boosting_model = 0
        self.special_objective = special_objective
        self.early_stopping_rounds = early_stopping_rounds
        self.fixed = True
        self.spherical = False
        
    def _build_wi(self, i, bw):

        try:
            wi = Kernel(i, self.coords, bw, fixed=self.fixed,
                        function=self.kernel, points=self.points,
                        spherical=self.spherical).kernel
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi        
    
        
    def fit_local(self,i):
        """
        Local fitting at location i.
        """
        wi = self._build_wi(i, self.bw).reshape(-1, 1)  #local spatial weights
        weighted_loss_obj = Geographicaly_Weighted_Loss(kernel=self.kernel,coords=self.coords,bw=self.bw,wi=wi)
        
        self.regression_model = xgb.train(self.para_dict, self.dtrain, self.num_round, self.eval_list,
                                            obj=weighted_loss_obj.weighted_loss, 
                                            verbose_eval=False, early_stopping_rounds=self.early_stopping_rounds)

        
        return self.regression_model



        
        
    def fit(self):
       #The Training Function

       
        if self.special_objective is None:
            # get the parameter list
            self.para_dict = {'max_depth': self.max_depth,
                              'colsample_bytree' : 0.35,
                              'learning_rate' : 0.15,
                              'alpha': 10,
                              'eta': self.eta,
                              'verbosity': self.verbosity,
                              'objective': self.objective_func,
                              'eval_metric': self.eval_metric,
                              'booster': self.booster}
        else:
            # get the parameter list, without stating the objective function
            self.para_dict = {'max_depth': self.max_depth,
                              'colsample_bytree' : 0.3,
                              'learning_rate' : 0.2,
                              'alpha': 10,
                              'eta': self.eta,
                              'verbosity': self.verbosity,
                              'eval_metric': self.eval_metric,
                              'booster': self.booster}
            
        if  self.kernel is None:
            self.kernel="Gaussian"
        
        # make sure data is in [nData * nSample] format
        # check if data length is the same
        if self.data_x.shape[0] != self.data_y.shape[0]:
            raise ValueError('The numbner of instances for x and y data should be the same!')
       
        # data_x is in [nData*nDim]
        nData = self.data_x.shape[0]
        #nData The rows of the data
        nDim = self.data_x.shape[1]
        
        self.dtrain = xgb.DMatrix(data=self.data_x, label=self.data_y)
        
        if self.special_objective is None:
            # fit the Normal Regressor
            #print("Normal Xgboost")
            self.regression_model = xgb.train(self.para_dict, self.dtrain, self.num_round, self.eval_list,
                                              verbose_eval=False, early_stopping_rounds=self.early_stopping_rounds)
       
        elif self.special_objective == 'weighted':
            # construct the object with imbalanced alpha value
            #print("Weighted Xgboost")
            #weighted_loss_obj = Geographicaly_Weighted_Loss(kernel=self.kernel,coords=self.coords,bw=self.bw)
            # fit the classfifier
            if self.points is None:
                m = self.data_y.shape[0]
            else:
                m = self.points.shape[0]
            result = self.fit_local(10)  #sequential

            
        

            
    def predict(self, data_x, y=None):
        # matrixilize
        if y is not None:
            try:
                dtest = xgb.DMatrix(data_x, label=y)
            except:
                raise ValueError('Test data invalid!')
        else:
            dtest = xgb.DMatrix(data_x)

        prediction_output = self.regression_model.predict(dtest)
        

        return prediction_output
