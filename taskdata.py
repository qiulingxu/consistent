import torch as T
from .base import TaskDataTransform, EvalBase
from .utils import assert_keys_in_dict, MAGIC, debug
from . import utils

from torch.utils.data import random_split
from abc import ABC, abstractmethod
import math
import copy

DEBUG = False

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

    def fill_evaluator(self, evaluator: EvalBase):
        return super().fill_evaluator(evaluator)

    @abstractmethod
    def _split_data(self):
        return True

    @abstractmethod
    def get_full_data():
        return []

class SeqTaskData(TaskDataTransform):
    def __init__(self, dataset, parameter):
        self.dataset = dataset
        self.parameter = parameter
        self._proc_parameter(parameter)
        self.gen_data_plan()

    def datafold(self,):
        self.split_fold = 10
        self.split_num = [2,1,7]

    def gen_data_plan(self):
        self.data_plan = {}
        for dp in self.dataset:
            idx = self._assign_elem_id(dp)
            if idx not in self.data_plan:
                self.data_plan[idx] = []
            self.data_plan[idx].append(dp)
        self.fill_evaluator("pertask_")
        self._post_process()
        self.split_data()
        #for k in self.data_plan.keys():
        #    self.data_plan[k] = T.stack(self.data_plan[k],dim=0)
    def _post_process(self):
        self.order = list(sorted(self.data_plan.keys()))
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
        split_num = copy.copy(self.split_num)
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

    def fill_evaluator(self, evaluator: EvalBase, prefix=""):
        for k in self.order:
            name = prefix+str(k)
            test, val, train = self._split_data(self.data_plan[k]) 
            for suffix, data in (("test",test), ("val",val), ("train", train)):
                evaluator.add_data(name+suffix, data)


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
        return True
    def _gen_task_mask(self):
        self.task_classes = {}
        for i in range(self.segments):
            for j in range(self.class_num):
                if self.labelmap[j] == i:
                    self.task_classes[i].append(j)
    def _assign_elem_id(self, dp):
        _, label = dp
        return self.labelmap[label]

class IncrementalClassificationData(ClassificationTaskData):
    def _post_process(self):
        self.order = list(sorted(self.data_plan.keys()))
        self.comparison = []
        for i in range(1,self.segments):
            self.comparison.append((i-1, i))
            self.data_plan[i].extend(self.data_plan[i-1])
            self.task_classes[i] = self.task_classes[i-1] + self.task_classes[i]
        if debug and DEBUG:
            print(i,self.data_plan[i][:10])

class CombineClassificationData(ClassificationTaskData):
    def _post_process(self):
        self.combine_key = self.segments
        self.order = list(sorted(self.data_plan.keys()))
        self.data_plan[self.combine_key] = []
        self.order.append(self.combine_key)
        self.comparison = []
        for i in range(0,self.segments):
            self.comparison.append((i, self.combine_key))
            self.data_plan[self.combine_key].extend(self.data_plan[i])
            self.task_classes[self.combine_key] += self.task_classes[i]