import torch as T
from .base import TaskDataTransform
from .utils import assert_keys_in_dict, MAGIC
from . import utils

from torch.utils.data import random_split
from abc import ABC, abstractmethod
import math

class NoTask(TaskDataTransform):
    def __init__(self, dataset, parameter):
        self.dataset = dataset
        self.parameter = parameter
        self.order = []
        self.gen_data_plan(parameter)

    def gen_data_plan(self, parameter):
        self.data_plan = {"0":self.dataset}

    def len(self,):
        return 1

    def get_plan(self):
        return self.data_plan

class SeqTask(TaskDataTransform):
    def __init__(self, dataset, parameter):
        self.dataset = dataset
        self.parameter = parameter
        self._proc_parameter(parameter)
        self.gen_data_plan()

    def gen_data_plan(self):
        self.data_plan = {}
        for dp in self.dataset:
            idx = self._assign_elem_id(dp)
            if idx not in self.data_plan:
                self.data_plan[idx] = []
            self.data_plan[idx].append(dp)
        self._post_process()
        self._split_data()
        #for k in self.data_plan.keys():
        #    self.data_plan[k] = T.stack(self.data_plan[k],dim=0)
    def _post_process(self):
        self.order = list(sorted(self.data_plan.keys()))
        self.comparison = []
        for i in range(1, self.len()):
            self.comparison.append((self.order[i-1], self.order[i]))
    
    def _split_data(self,):
        self.split_fold = 10
        self.data_plan_train = {}
        self.data_plan_test = {}
        self.data_plan_val = {}
        for k in self.order:
            l = len(self.data_plan[k])
            self.data_plan_train[k] = []
            self.data_plan_test[k] = []
            self.data_plan_val[k] = []
            for idx, dp in enumerate(l):
                slice = idx%self.split_fold 
                if slice in [0,1]:
                    self.data_plan_test[k].append(dp)
                elif slice in [2]:
                    self.data_plan_val[k].append(dp)
                else:
                    self.data_plan_train[k].append(dp)

    def _proc_parameter(self, parameter):
        return None

    def _assign_elem_id(self, dp):
        return 0

    def len(self,):
        return self.len(self.data_plan)
    
    def len_task(self, idx):
        return list(self.data_plan[idx].shape)[0]

    def get_plan(self):
        return self.data_plan        

class IncrementalClassification(SeqTask):
    def _proc_parameter(self, parameter):
        if "labelmap" not in parameter:
            assert_keys_in_dict("segments", parameter)
            self.segments = parameter["segments"]
            self.class_num = utils.config["CLASS_NUM"]
            self.class_per_seg = math.ceil(self.class_num / self.segments)
            def map_to_task(label):
                return label // self.class_per_seg
            for i in range(self.class_num):
                self.labelmap[i] = map_to_task(i)
        else:
            self.labelmap = parameter["labelmap"]
            self.segments = len(self.labelmap)
        return True
    def _assign_elem_id(self, dp):
        _, label = dp
        return self.labelmap[label]
    def _post_process(self):
        self.order = list(sorted(self.data_plan.keys()))
        self.comparison = []
        for i in range(1,self.segments):
            self.comparison.append((i-1, i))
            self.data_plan[i].extend(self.data_plan[i-1])


class CombineClassification(IncrementalClassification):
    def _post_process(self):
        self.combine_key = self.segments
        self.order = list(sorted(self.data_plan.keys()))
        self.data_plan[self.combine_key] = []
        self.order.append(self.combine_key)
        self.comparison = []
        for i in range(0,self.segments):
            self.comparison.append((i, self.combine_key))
            self.data_plan[self.combine_key].extend(self.data_plan[i])
