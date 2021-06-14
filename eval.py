from typing import Dict
import torch as T
import torch.nn as nn
import numpy as np

from typing import Any, List, Dict, Tuple
from .base import *
from .utils import PytorchModeWrap as PMW
from .taskdata import SeqTaskData
from .evaldata import FixDataMemoryBatchClassification as MBC_FD

def order_condition(step, order):
    control, val = order
    if control == "from":
        return step>=val
    else:
        assert False, "Implement control order {}".format(control)

class EvalProgressPerSample(EvalBase):
    def __init__(self, metric: nn.Module, device, max_step = 10000, **karg):
        super(EvalProgressPerSample, self).__init__(metric, device, max_step)
        self.metric = metric.to(device)
        self.data = {} # type: Dict [Any, FixData]
        self.len = {} # type: Dict [Any, int]
        self.hist_version :Dict[Any, np.ndarray] = {} 
        self.max_step = max_step
        self.curr_step = 0
        self.device = device
        self.names: List[Any] = [] # type List[Any] 
        self.orders:Dict[Any, Tuple[str, Any]] =  {} # type 

        self.process_parameters(**karg)

    def process_parameters(self,**karg):
        pass

    def add_data(self, name:str, data:Iterable[Any], batch_size:int, **kargs):
        fd = MBC_FD(batch_size=batch_size)
        fd.feed_data(data)
        self.add_fix_data(name, fd, **kargs)
    
    def add_fix_data(self, name: str, data: FixData, count_from=0):
        assert name not in self.names, "duplicate name"
        self.names.append(name)
        self.data[name] = data
        len = data.len_element() 
        self.len[name] = len
        self.hist_version[name] = np.zeros(shape=(len, self.max_step), dtype = np.float32)
        self.orders[name] = ("from",count_from)

    def eval(self, model):
        model = model.to(self.device)
        with PMW(model, training=False):
            for name in self.names:
                for dp in self.data[name].get_iterator():
                    input = [i.to(self.device) for i in dp["input"]]
                    dp = {k: i.to(self.device) for k, i in dp.items() if k != "input"}
                    output = model(*input)
                    score = self.metric(output, dp, model)
                    idx = dp["idx"]
                    # in batch idx
                    for ib_idx, e_idx in enumerate(idx):
                        e_idx = int(e_idx)
                        self.hist_version[name][e_idx, self.curr_step] = score[ib_idx]
        self.curr_step += 1
    def measure(self):
        rst = {}
        for name in self.names:
            rst[name] = self._measure([name])
        rst["total"] = self._measure(self.names)


    def _measure(self, keys):
        cnt = 0
        tot_cnt = 0
        for name in self.keys:
            hist = self.hist_version[name]
            length = self.len[name]
            tot_cnt += length
            for i in range(length):

                f=None
                for j in range(0,self.curr_step):
                    if order_condition(j):
                        if f is None:
                            f = hist[i,j]
                        else:
                            if f < hist[i,j]:
                                #print("sample %d" % i)
                                cnt += 1
                                break
                        f = max(f, hist[i,j])
        rst = {"inconsist":cnt*1.0/tot_cnt, "cnt":tot_cnt}
        return rst

