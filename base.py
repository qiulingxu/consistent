from abc import ABC, abstractmethod
from collections.abc import Callable

from typing import Union, Any, Iterable
import torch as T
import torch.nn as nn
import numpy as np

class FixData(ABC):
    @abstractmethod
    def feed_data(self, *arg, **karg):
        print("feed_data")
    @abstractmethod
    def len(self):
        """ length for iterator"""
        return 100
    @abstractmethod
    def len_element(self):
        return 100
    @abstractmethod
    def get_data_by_id(self, id):
        return np.arange(start=0, stop=32),{"x":np.zeros((32,10,1))}
    def get_iterator(self):
        for i in range(self.len()):
            yield self.get_data_by_id(i)

class Task(ABC):
    @abstractmethod
    def __init__(self, *arg, **karg):
        pass

    @abstractmethod
    def get_name(self,):
        return "no_task"

    @abstractmethod
    def process_data(self, dataset):
        return dataset

    @abstractmethod
    def controlled_train(self, Model, dataset):
        pass

class EvalBase(object):
    @abstractmethod
    def __init__(self, metric: nn.Module, device, max_step):
        pass

    @abstractmethod
    def add_data(self, name:str, data:Iterable[Any]):
        pass

    @abstractmethod
    def add_fix_data(self, name:str, data:FixData):
        pass

    @abstractmethod
    def eval(self, model):
        pass
    
    @abstractmethod
    def measure(self,):
        return {}

class TaskDataTransform(ABC):
    def __init__(self, dataset, parameter):
        self.dataset = dataset
        self.parameter = parameter
        self.gen_data_plan(parameter)

    @abstractmethod
    def gen_data_plan(self, parameter):
        self.data_plan = {"0":self.dataset}

    @abstractmethod
    def len(self,):
        return 0

    def get_plan(self):
        return self.data_plan

    @abstractmethod
    def fill_evaluator(self,evaluator:EvalBase):
        return True
