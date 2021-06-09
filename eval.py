import torch as T
import torch.nn as nn
import numpy as np

from .base import *
from .utils import PytorchModeWrap as PMW

class EvalProgressPerSample():
    def __init__(self, data : FixData, metric: nn.Module, device, max_step = 10000):
        super(EvalProgressPerSample, self).__init__()
        self.metric = metric.to(device)
        self.data = data
        self.max_step = max_step
        self.len = self.data.len_element() 
        self.hist_version= np.zeros(shape=(self.len, self.max_step), dtype = bool)
        self.curr_step = 0
        self.device = device
    def eval(self, model):
        model = model.to(self.device)
        with PMW(model, training=False):
            for dp in self.data.get_iterator():
                input = [i.to(self.device) for i in dp["input"]]
                dp = {k: i.to(self.device) for k, i in dp.items() if k != "input"}
                output = model(*input)
                score = self.metric(output, dp)
                idx = dp["idx"]
                # in batch idx
                for ib_idx, e_idx in enumerate(idx):
                    e_idx = int(e_idx)
                    self.hist_version[e_idx, self.curr_step] = score[ib_idx]
        self.curr_step += 1
    def inconsistency(self):
        cnt = 0

        for i in range(self.len):
            f=self.hist_version[i, 0]
            for j in range(1,self.curr_step):
                
                if f < self.hist_version[i,j]:
                    #print("sample %d" % i)
                    cnt += 1
                    break
                f = max(f, self.hist_version[i,j])
        #print()
        return cnt*1.0/self.len
        