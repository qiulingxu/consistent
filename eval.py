from typing import Dict
import torch as T
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple, Optional
import json

from .base import *
from .utils import PytorchModeWrap as PMW
from .evaldata import FixDataMemoryBatchClassification as MBC_FD

def order_condition(step, order):
    control, val = order
    if control == "from":
        return step>=val
    elif control == "in":
        return step in val
    elif control == "all":
        return True
    else:
        assert False, "Implement control order {}".format(control)

EPS = 1e-5

INCON_TOT_TYPE = "Correct"
assert INCON_TOT_TYPE in ["Correct", "All"]
class EvalProgressPerSample(EvalBase):
    def __init__(self, device, max_step = 50, **karg):
        super().__init__(device, max_step, **karg)
        self.metrics = {} # type: Dict[str, nn.Module]
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

    def add_data(self, name:str, data:Dataset, metric: nn.Module, batch_size:int,  **kargs): #Iterable[Any]
        assert isinstance(data, Dataset), type(data)
        fd = MBC_FD(batch_size=batch_size)
        fd.feed_data(data)
        self.add_fix_data(name, fd, metric, **kargs)
    
    def add_fix_data(self, name: str, data: FixData, metric: nn.Module, order: Optional[Tuple[str, Any]] = None):
        assert name not in self.names, "duplicate name"
        self.names.append(name)
        self.metrics[name] = metric
        self.data[name] = data
        self.metrics[name] = metric
        len = data.len_element() 
        self.len[name] = len
        self.hist_version[name] = np.zeros(shape=(len, self.max_step), dtype = np.float32)
        if order is None:
            self.orders[name] = ("from",0)
        else:
            self.orders[name] = order

    def find_match_model(self, models, key):
        ks = models.keys()
        result = None
        for k in ks:
            if key.find(k)>=0: 
                if result is None:
                    result = k
                else:
                    assert False, "Find duplicate mathcing models {} and {} for task {}.".format(result, k, key)
        assert result is not None, "Find no mathcing models {} and {} for task {}.".format(result, k, key)
        return result
    def eval(self, task2models, step = None):
        """We have task name and dataseet names.
            the dataset name must contain exactly one task name to build correspondence
        """
        if step is not None:
            use_step = step 
        else:
            use_step = self.curr_step
        for name in self.names:
            key = self.find_match_model(task2models, name)
            if order_condition(use_step, self.orders[name]):
                if key is None:
                    assert False, "You specify a order {} but did not \
                        provide the model during evaluation {}".format(str(self.orders[name]),name) 
            else:#if key is None:
                """This comparison is not required in order, thus skip it"""
                continue
            model = task2models[key]
            if isinstance(model, dict):
                curr_model = model[name].to(self.device)
            else:
                curr_model = model.to(self.device)
            with PMW(curr_model, training=False):
                metric = self.metrics[name].to(self.device)
                for dp in self.data[name].get_iterator():
                    input = [i.to(self.device) for i in dp["input"]]
                    dp = {k: i.to(self.device) for k, i in dp.items() if k != "input"}
                    output = curr_model(*input)
                    score = metric(output, dp, curr_model)
                    idx = dp["idx"]
                    # in batch idx
                    for ib_idx, e_idx in enumerate(idx):
                        e_idx = int(e_idx)
                        self.hist_version[name][e_idx, use_step] = score[ib_idx]
        if step is not None:
            self.curr_step = max(self.curr_step, step+1)
        else:
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
        full_score = []
        for compare_1 in range(l_steps):
            step_1 = valid_step[compare_1]
            for compare_2 in range(compare_1+1, l_steps):
                step_2 = valid_step[compare_2]
                cnt = 0
                for l in range(length):
                    if hist[l,step_1]>hist[l, step_2] + EPS:
                        cnt += 1
                full_score.append({"compare": (step_1, step_2), "consistency": cnt * 1.0 / length})
        return {"inconsist_pairwise":full_score, "cnt":length, "index":valid_step}

    def _measure(self, names):
        cnt = 0
        tot_cnt = 0
        for name in names:
            hist = self.hist_version[name]
            length = self.len[name]
            #tot_cnt += length
            valid_step = self._get_valid_step(name)
            l_steps = len(valid_step)
            # altogether comparison
            for l in range(length):
                prev=None
                for j in range(l_steps):
                    step = valid_step[j]
                    curr = hist[l, step]
                    if prev is None:
                        if INCON_TOT_TYPE == "Correct" and curr!= 1.0:
                            break
                        tot_cnt += 1
                    elif prev > curr + EPS:
                        #print("sample %d" % i)
                        cnt += 1
                        break
                    prev = curr
        rst = {"inconsist_tot":cnt*1.0/tot_cnt, "cnt":tot_cnt, "index":valid_step}
        return rst

    def save(self, filename):
        _js_file = filename + "_measure.json"
        hist_file = filename + "_hist.npy"
        _measure = self.measure()
        _config = {"orders":self.orders, "len": self.len, "curr_step":self.curr_step, "names": self.names}
        _measure.update(_config)
        with open(_js_file, "w") as f:
            f.write(json.dumps(_measure))
        np.savez(hist_file,**self.hist_version)
    
    def load(self,filename):
        _js_file = filename + "_measure.json"
        hist_file = filename + "_hist.npy"
        with open(_js_file, "r") as f:
            _measure = json.loads(f.read())
        self.orders = _measure["orders"]
        self.len = _measure["len"]
        self.curr_step = _measure["curr_step"]
        self.names = _measure["names"]
        self.hist_version = np.load(hist_file + ".npz")
    
    def set_order(self, option, val=None):
        print(option)
        assert option in ["sequential", "set"]
        
        for k,v in self.orders.items():
            _k = k[k.find("Task"):].split("_")
            task_id = int(_k[1])
            if option == "sequential":
                self.orders[k] = ("from", task_id)
            else:
                self.orders[k] = ("in", val)
            #print(k,v)


    @abstractmethod
    def _define_acc(self, score: float) -> float:
        assert False
        return 1

    
class EvalProgressPerSampleClassification(EvalProgressPerSample):
    def _define_acc(self, score):
        return score

class EvalReader(EvalProgressPerSampleClassification):
    def __init__(self, max_step = 200, **karg):
        super().__init__(device="cpu", max_step = max_step)
