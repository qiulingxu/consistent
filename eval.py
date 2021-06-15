from typing import Dict
import torch as T
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple, Optional
import json

from .base import *
from .utils import PytorchModeWrap as PMW
from .taskdata import SeqTaskData
from .evaldata import FixDataMemoryBatchClassification as MBC_FD

def order_condition(step, order):
    control, val = order
    if control == "from":
        return step>=val
    elif control == "in":
        return step in val
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
    
    def add_fix_data(self, name: str, data: FixData, order: Optional[Tuple[str, Any]] = None):
        assert name not in self.names, "duplicate name"
        self.names.append(name)
        self.data[name] = data
        len = data.len_element() 
        self.len[name] = len
        self.hist_version[name] = np.zeros(shape=(len, self.max_step), dtype = np.float32)
        if order is None:
            self.orders[name] = ("from",0)
        else:
            self.orders[name] = order

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
            rst[name+"_pairwise_ic"] = self._measure_pairwise(name)
            rst[name+"_acc"] = self._measure_acc(name)
            rst[name+"_ic"] = self._measure([name])
        rst["total_ic"] = self._measure(self.names)
        return rst

    def _get_valid_step(self, name):
        valid_step = []
        for i in range(0, self.curr_step):
            if order_condition(i, self.orders[name]):
                valid_step.append(i)
        return valid_step

    def _measure_acc(self, name):
        cnt = 0
        hist = self.hist_version[name]
        length = self.len[name]
        valid_step = self._get_valid_step(name)
        l_steps = len(valid_step)
        # altogether comparison
        f=None
        accs = []
        for j in range(l_steps):
            cnt = 0
            for l in range(length):
                step = valid_step[j]
                _h = hist[l, step]
                cnt += self._define_acc(_h)
            accs.append(cnt*1.0/length )
                    
        rst = {"acc_list":accs, "cnt":length, "index":valid_step}
        return rst

    def _measure_pairwise(self, name):
        hist = self.hist_version[name]
        length = self.len[name]
        
        valid_step = self._get_valid_step(name)
        l_steps = len(valid_step)
        # pairwise comparison
        full_score = {}
        for compare_1 in range(l_steps):
            step_1 = valid_step[compare_1]
            for compare_2 in range(compare_1+1, l_steps):
                step_2 = valid_step[compare_2]
                cnt = 0
                for l in range(length):
                    if hist[l,step_1]>hist[l, step_2]:
                        cnt += 1
                full_score[(compare_1, compare_2)] = cnt * 1.0 / length
        return {"inconsist_pairwise":full_score, "cnt":length, "index":valid_step}

    def _measure(self, names):
        cnt = 0
        tot_cnt = 0
        for name in names:
            hist = self.hist_version[name]
            length = self.len[name]
            tot_cnt += length
            valid_step = self._get_valid_step(name)
            l_steps = len(valid_step)
            # altogether comparison
            f=None
            for l in range(length):
                for j in range(l_steps):
                    step = valid_step[j]
                    _h = hist[l, step]
                    if f is None:
                        f = _h
                    elif f < _h:
                        #print("sample %d" % i)
                        cnt += 1
                        break
                    f = max(f, _h)
        rst = {"inconsist_tot":cnt*1.0/tot_cnt, "cnt":tot_cnt, "index":valid_step}
        return rst

    def save(self, filename):
        _js_file = filename + "_measure.json"
        hist_file = filename + "_hist.npy"
        _measure = self.measure()
        _config = {"orders":self.orders, "len": self.len, "curr_step":self.curr_step, "names": self.names}
        _measure.update(_config)
        with open(filename, "w") as f:
            f.write(json.dumps(_measure))
        np.savez(hist_file,**self.hist_version)
        
    @abstractmethod
    def _define_acc(self, score):
        return 1

class EvalProgressPerSampleClassification(EvalProgressPerSample):
    def _define_acc(self, score):
        return score