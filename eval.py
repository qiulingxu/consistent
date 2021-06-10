import torch as T
import torch.nn as nn
import numpy as np

from .base import *
from .utils import PytorchModeWrap as PMW



class EvalProgressPerSample(EvalBase):
    def __init__(self, metric: nn.Module, device, max_step = 10000):
        super(EvalProgressPerSample, self).__init__()
        self.metric = metric.to(device)
        self.data = {}
        self.len = {}
        self.hist_version = {}
        self.max_step = max_step
        self.curr_step = 0
        self.device = device
        self.names = []

    def add_data(self, name: str, data: FixData):
        assert name not in self.key, "duplicate name"
        self.names.append(name)
        self.data[name] = data
        len = data.len_element() 
        self.len[name] = len
        self.hist_version[name] = np.zeros(shape=(len, self.max_step), dtype = np.float32)
        
    def eval(self, model):
        model = model.to(self.device)
        with PMW(model, training=False):
            for name in self.names:
                for dp in self.data[name].get_iterator():
                    input = [i.to(self.device) for i in dp["input"]]
                    dp = {k: i.to(self.device) for k, i in dp.items() if k != "input"}
                    output = model(*input)
                    score = self.metric(output, dp)
                    idx = dp["idx"]
                    # in batch idx
                    for ib_idx, e_idx in enumerate(idx):
                        e_idx = int(e_idx)
                        self.hist_version[name][e_idx, self.curr_step] = score[ib_idx]
        self.curr_step += 1
    def measure(self):
        cnt = 0
        rst = {}
        for name in self.names:
            hist = self.hist_version[name]
            for i in range(self.len[name]):
                f=hist[i, 0]
                for j in range(1,self.curr_step):
                    
                    if f < hist[i,j]:
                        #print("sample %d" % i)
                        cnt += 1
                        break
                    f = max(f, hist[i,j])
            rst[name] = {"inconsist":cnt*1.0/self.len}
        #print()
        return rst