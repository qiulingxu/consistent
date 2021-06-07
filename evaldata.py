import torch as T
import torch.nn as nn
import numpy as np
import math

from .base import *

TORCH = 1
NUMPY = 2
VERSION = TORCH
nptype = np.ndarray

class FixDataMemoryBatch(FixData):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def feed_data(self, data):
        assert isinstance(data, dict)
        self.l = None
        for k in data.keys():
            if VERSION == NUMPY:
                if T.is_tensor(data[k]):
                    data[k] = data[k].numpy()
                assert isinstance(data[k], nptype)
                if self.l is None:
                    self.l = data[k].shape[0]
                else:
                    assert self.l == data[k].shape[0]
            elif VERSION == TORCH:
                if isinstance(data[k], nptype):
                    data[k] = T.from_numpy(data[k])
                assert T.is_tensor(data[k])
                curr_l = list(data[k].shape)[0]
                if self.l is None:
                    self.l = curr_l
                else:
                    assert self.l == curr_l
            self.data = data
        self.l_batch = math.ceil(self.l / self.batch_size) 
    
    def len(self):
        return self.l_batch
    
    def len_element(self):
        return self.l

    def get_data_by_id(self, id):
        ret = {}
        assert id<self.len()
        for k in self.data.keys():
            ret[k] = self.data[k][self.batch_size*id:self.batch_size*(id+1)] 
        if VERSION == NUMPY:
            l = ret[k].shape[0]
            ret["idx"] = np.arange(start=self.batch_size*id, end=self.batch_size*id+l)
            return ret
        elif VERSION == TORCH:
            l = list(ret[k].shape)[0]
            ret["idx"] = T.arange(start=self.batch_size*id, end=self.batch_size*id+l)
            return ret

class FixDataMemoryBatchClassification(FixDataMemoryBatch):
    """ Assumes we have the data from pytorch dataset, where it is a iterator with form dp[0], dp[1].
        For classification problem, we assume it has (sample, label)"""
    def feed_data(self, data):
        assert VERSION in [NUMPY, TORCH]
        self.x = []
        self.y = []
        for dp in data:
            #print(dp)
            e_x, e_y = dp
            if VERSION == NUMPY:
                if isinstance(e_x, T.tensor):
                    e_x = e_x.numpy()
                if isinstance(e_y, T.tensor):
                    e_y = e_y.numpy()
            elif VERSION == TORCH:
                if not T.is_tensor(e_x):
                    e_x = T.tensor(e_x)#T.from_numpy(e_x)
                if not T.is_tensor(e_y):
                    e_y = T.tensor(e_y)#T.from_numpy(e_y)              
            self.x.append(e_x)
            self.y.append(e_y)
        if VERSION == NUMPY:
            self.x = np.stack(self.x, axis=0)
            self.y = np.stack(self.y, axis=0)
        elif VERSION == TORCH:
            print(len(self.y))
            self.x = T.stack(self.x, dim=0)
            self.y = T.stack(self.y, dim=0)            
        return super().feed_data({"x": self.x , "y": self.y})
    def get_data_by_id(self, id):
        ret = super().get_data_by_id(id)
        ret["input"] = [ret["x"]]
        return ret






if __name__=="__main__":
    pass