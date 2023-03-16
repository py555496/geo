import numpy as np
import xgboost as xgb
from typing import Tuple
from kernels import *

class Geographicaly_Weighted_Loss:
    '''
    The class of binary cross entropy loss, allows the users to change the weight parameter
    '''
    def __init__(self,kernel,coords,bw,wi):
        '''
        :param imbalance_alpha: the imbalanced \alpha value for the minority class (label as '1')
        '''
        self.kernel=kernel
        self.coords=coords
        self.bw=bw
        self.wi=wi
        self.fixed=True
        self.points=None
        self.spherical=False



    def weighted_loss(self, pred, dtrain):
        
        
        label = np.array(dtrain.get_label()).reshape(-1,1)
       # retrieve data from dtrain matrix
        pred = np.array(pred).reshape(-1,1)
        wi=np.array(self.wi).reshape(-1,1)

        
        # gradient
        grad =np.array((pred-label)*wi)
        
        hess = (np.array(np.ones(len(pred),dtype=np.float32)).reshape(-1,1))*wi

        

        return grad, hess

        
  

