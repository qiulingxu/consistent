from abc import ABC, abstractmethod
from collections.abc import Callable

from typing import Union, Any, Iterable, Dict, Tuple, List, Sized
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
    def __init__(self, device, max_step):
        pass

    @abstractmethod
    def add_data(self, name:str, data:Iterable[Any], metric: nn.Module, order:Tuple[str, Any], **karg):
        pass

    @abstractmethod
    def add_fix_data(self, name:str, data:FixData, metric: nn.Module, order:Tuple[str, Any]):
        pass

    @abstractmethod
    def eval(self, models: Dict[str, nn.Module]):
        pass
    
    @abstractmethod
    def measure(self,):
        return {}

class TaskDataTransform(ABC):
    def __init__(self, dataset, metric, parameter):
        self.dataset = dataset
        self.parameter = parameter
        self.metric = metric
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
    def fill_evaluator(self,evaluator:EvalBase, prefix=""):
        return True

class MultiTaskDataTransform(ABC):
    @abstractmethod
    def add_task_data(self, taskdata:TaskDataTransform, taskname:str):
        pass

    @abstractmethod
    def list_tasks(self) -> List[str]:
        return []

    @abstractmethod
    def get_task_data(self, taskname:str, order:Any, fold:str) -> Sized:
        return []

    @abstractmethod
    def get_metric(self, taskname:str) -> nn.Module:
        return None

    @abstractmethod
    def get_task_compare(self, taskname:str, order:Any) -> Sized:
        return []

    @abstractmethod
    def fill_evaluator(self, evaluator: EvalBase, prefix=""):
        return


class ClassificationModule(nn.Module):

    @abstractmethod
    def get_linear(self):
        return None

    @abstractmethod
    def sublabels(self, labels):
        pass

    @abstractmethod
    def process_labels(self, labels):
        return labels

    @abstractmethod
    def process_output(self, output):
        return output