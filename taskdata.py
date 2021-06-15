import torch as T
from typing import List, Any, Dict, Tuple
from torch.utils.data import random_split
from abc import ABC, abstractmethod
import math
import copy

from .base import TaskDataTransform, EvalBase
from .utils import assert_keys_in_dict, MAGIC, debug
from . import utils

DEBUG = False

class NoTask(TaskDataTransform):
    def __init__(self, dataset, parameter, evaluator:EvalBase):
        self.dataset = dataset
        self.parameter : Dict[str,Any] = parameter
        self.order: List[Any] = []
        self.evaluator = evaluator
        self.gen_data_plan(parameter)

    def gen_data_plan(self, parameter):
        self.data_plan = {"0":self.dataset}

    def len(self,):
        return 1

    def get_plan(self):
        return self.data_plan

    def fill_evaluator(self, evaluator: EvalBase):
        return super().fill_evaluator(evaluator)

    @abstractmethod
    def _split_data(self):
        return True

    @abstractmethod
    def get_full_data(self,):
        return []

class SeqTaskData(TaskDataTransform):
    def __init__(self, dataset, parameter, evaluator:EvalBase):
        self.dataset = dataset
        self.parameter = parameter
        self.evaluator = evaluator
        self._proc_parameter(parameter)
        self.gen_data_plan()

    def define_datafold(self,):
        """Overide this function to redefine the datafold rule"""
        self.split_fold = 10
        self.split_num = [2,1,7]
        self.slices = len(self.split_num)

    def define_order(self):
        """Overide this function to redefine the order"""
        self.order = list(sorted(self.data_plan.keys()))

    def gen_data_plan(self):
        self.define_datafold()
        self.data_plan = {}
        for dp in self.dataset:
            idx = self._assign_elem_id(dp)
            if idx not in self.data_plan:
                self.data_plan[idx] = []
            self.data_plan[idx].append(dp)
        self.define_order()
        self.fill_evaluator(self.evaluator,"pertask_")
        self._post_process()
        self.split_data()
        #for k in self.data_plan.keys():
        #    self.data_plan[k] = T.stack(self.data_plan[k],dim=0)
    def _post_process(self):
        self.comparison = []
        for i in range(1, self.len()):
            self.comparison.append((self.order[i-1], self.order[i]))
    
    def split_data(self,):
        self.data_plan_train = {}
        self.data_plan_test = {}
        self.data_plan_val = {}
        for k in self.order:
            l = len(self.data_plan[k])
            self.data_plan_test[k],self.data_plan_val[k],self.data_plan_train[k] \
                = self._split_data(self.data_plan[k])


    def _split_data(self,data):
        split_num = copy.deepcopy(self.split_num)
        for i in range(1, self.slices):
            split_num[i] += split_num[i-1]
        assert split_num[-1] == self.split_fold
        
        lsts = [[] for i in range(self.slices)]
        for idx, dp in enumerate(data):
            slice = idx%self.split_fold 
            for i in range(self.slices):
                if slice < split_num[i]:
                    lsts[i].append(dp)
                    break
        return lsts

    def fill_evaluator(self, evaluator: EvalBase, prefix=""):
        for k in self.order:
            name = prefix+str(k)
            test, val, train = self._split_data(self.data_plan[k]) 
            for suffix, data in (("test",test), ("val",val), ("train", train)):
                evaluator.add_data(name+suffix, data, \
                    batch_size=self.batch_size, \
                    order=self._define_order(k))

    @abstractmethod
    def _define_order(self, k:str) -> Tuple[str, Any]:
        return ("Order_Type", "VALUE")

    def _proc_parameter(self, parameter):
        self.batch_size = parameter["batch_size"]
        return None

    def _assign_elem_id(self, dp):
        return 0

    def len(self,):
        return self.len(self.data_plan)
    
    def len_task(self, idx):
        return list(self.data_plan[idx].shape)[0]

    def get_plan(self):
        return self.data_plan       

class ClassificationTaskData(SeqTaskData):
    def _proc_parameter(self, parameter):
        if "labelmap" not in parameter:
            assert_keys_in_dict(["segments"], parameter)
            self.segments = parameter["segments"]
            self.class_num = utils.config["CLASS_NUM"]
            self.class_per_seg = math.ceil(self.class_num / self.segments)
            def map_to_task(label):
                return label // self.class_per_seg
            self.labelmap = {}
            for i in range(self.class_num):
                self.labelmap[i] = map_to_task(i)
        else:
            self.labelmap = parameter["labelmap"]
            self.segments = len(set(self.labelmap.values()))
        self._gen_task_mask()
        return super()._proc_parameter(parameter)
        
    def _gen_task_mask(self):
        self.task_classes = {i:[] for i in range(self.segments)}

        for j in range(self.class_num):
            self.task_classes[self.labelmap[j]].append(j)
    def _assign_elem_id(self, dp):
        _, label = dp
        return self.labelmap[label]

class IncrementalDomainClassificationData(ClassificationTaskData):
    def _post_process(self):
        print("Into IncrementalClassificationData post process")
        self.comparison = []
        for i in range(1,self.segments):
            self.comparison.append((i-1, i))
            self.data_plan[i].extend(self.data_plan[i-1])
            self.task_classes[i] = self.task_classes[i-1] + self.task_classes[i]
        if debug and DEBUG:
            print(i,self.data_plan[i][:10])

    def _define_order(self, k):
        """ Each Task contains all information from before, but includes new ones. 
        Thus order is defined from current position
        """
        return ("from", self.order.index(k))

class CombineClassificationData(ClassificationTaskData):
    def _post_process(self):
        self.combine_key = self.segments
        self.data_plan[self.combine_key] = []
        self.order.append(self.combine_key)
        self.comparison = []
        for i in range(0,self.segments):
            self.comparison.append((i, self.combine_key))
            self.data_plan[self.combine_key].extend(self.data_plan[i])
            self.task_classes[self.combine_key] += self.task_classes[i]
    def _define_order(self, k):
        """ Each task is concurrently trained.
        Thus we define the order as 
        """
        combine_id = self.order.index(self.combine_key)
        return ("in", [self.order.index(k),combine_id])