import torch as T
import torch.nn as nn
from typing import List, Any, Dict, Tuple
from torch.utils.data import random_split
from abc import ABC, abstractmethod
import math
import copy
import random

from .base import TaskDataTransform, EvalBase, MultiTaskDataTransform
from .utils import assert_keys_in_dict, MAGIC, debug, log
from . import utils
from . import task

DEBUG = False

class NoTask(TaskDataTransform):
    def __init__(self, dataset, evaluator:EvalBase, metric:nn.Module, **parameter):
        self.dataset = dataset
        self.parameter : Dict[str,Any] = parameter
        self.order: List[Any] = []
        self.evaluator = evaluator
        self.metric = metric
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



class SeqTaskData(TaskDataTransform, MultiTaskDataTransform):
    def __init__(self, dataset, evaluator:EvalBase, metric:nn.Module, taskname:str = "Task",  **parameter):
        self.dataset = dataset
        self.taskname = taskname
        self.parameter = parameter
        self.evaluator = evaluator
        self.metric = metric
        self.do_shuffle = True
        self._proc_parameter(parameter)
        self.l = {}  #type: Dict[str, int]
        self.gen_data_plan()

        # simulate multi tasks 
        self.tasks ={}
        self.tasks[self.taskname] = self

    def list_tasks(self):
        return [self.taskname]

    def add_task_data(self, taskdata:TaskDataTransform, taskname:str):
        assert False, "Sequential task data only simulates the read interface for MultiTaskDataTransform"
        return

    def define_datafold(self,):
        """Overide this function to redefine the datafold rule"""
        self.split_fold = 10
        self.split_num = [2,1,7]
        self.slices = len(self.split_num)

    def define_order(self):
        """Overide this function to redefine the order"""
        self.order = list(sorted(self.data_plan.keys()))

    def get_data(self, order, fold):
        assert fold in ["train", "test", "val"]
        if fold == "train":
            return self.data_plan_train[order]
        elif fold == "test":
            return self.data_plan_test[order]
        elif fold == "val":
            return self.data_plan_val[order]

    def get_comparison(self,):
        return self.comparison

    def get_task_compare(self, taskname:str, order:Any):
        assert taskname == self.taskname
        compare_pairs = []         
        for compare_pair in self.get_comparison():   
            if compare_pair[-1] == order:
                compare_pairs.append(compare_pair[0])  
        return compare_pairs

    def get_task_data(self, taskname:str, order:Any, fold:str):
        assert taskname == self.taskname
        return self.get_data(order, fold)

    def gen_data_plan(self):
        self.define_datafold()
        if self.do_shuffle:
            self.shuffle_data()
        self.data_plan = {}
        for dpid, dp in enumerate(self.dataset):
            idx = self._assign_elem_id(dpid, dp)
            if idx not in self.data_plan:
                self.data_plan[idx] = []
            self.data_plan[idx].append(dp)
        self.define_order()
        self.fill_evaluator(self.evaluator,"perslice_")
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
            self.l[k] = l
            if l >= 10:
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
            name = "{}_{}_{}_".format(prefix,self.taskname,str(k))
            if len(self.data_plan[k]) >= 10:
                test, val, train = self._split_data(self.data_plan[k])   
                for suffix, data in (("test",test), ("val",val), ("train", train)):
                    evaluator.add_data(name+suffix, data, \
                        batch_size=self.batch_size, \
                        order=self._define_order(k),
                        metric = self.metric)
            else:
                log("The data plan for {} is less than 10 samples and discarded.".format(k))

    @abstractmethod
    def _define_order(self, k:str) -> Tuple[str, Any]:
        return ("Order_Type", "VALUE")

    def _proc_parameter(self, parameter):
        self.batch_size = parameter["batch_size"]
        return None

    def _assign_elem_id(self, dpid, dp):
        """datapoint id in whole dataset, and datapoint"""
        return 0

    def len(self,):
        return self.len(self.data_plan)
    
    def len_task(self, idx):
        return list(self.data_plan[idx].shape)[0]

    def get_plan(self):
        return self.data_plan     

    def shuffle_data(self,):
        random.seed(utils.MAGIC)
        self.dataset = list(self.dataset)
        random.shuffle(self.dataset)

class SeqTaskNaiveData(SeqTaskData):
    """Interface for multi task learning"""
    def _proc_parameter(self, parameter):
        assert "order" in parameter
        self.order = parameter["order"]
        self.task_classes = list(range(utils.config["CLASS_NUM"]))
        assert "task_number" in parameter 
        assert parameter["task_relation"] in ["concurrent", "parallel"]
        self.task_relation = parameter["task_relation"]
        self.task_number = parameter["task_number"]
        if self.task_relation == "concurrent":
            self.combine_key = self.task_number

    def _assign_elem_id(self, dpid, dp):
        return self.order

    def fill_evaluator(self, evaluator: EvalBase, prefix=""):
        k = self.order
        name = "{}_{}_{}_".format(prefix,self.taskname,str(k))
        test, val, train = self._split_data(self.data_plan[k]) 
        
        if self.l[k] >= 10:
            for suffix, data in (("test",test), ("val",val), ("train", train)):
                evaluator.add_data(name+suffix, data, \
                    batch_size=self.batch_size, \
                    order=self._define_order(k),
                    metric = self.metric)    

    def _define_order(self, k):
        if self.task_relation == "concurrent":
            return ("in", [self.order, self.combine_key])
        elif self.task_relation == "parallel":
            return ("from", self.order)
        else:
            assert False

class ClassificationTaskData(SeqTaskData):
    def _proc_parameter(self, parameter):
        if "task_classes" not in parameter:
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
        else:
            self.task_classes = parameter["task_classes"]
        if "to_mul_task" in parameter:
            self.to_mul_task = True
        else:
            self.to_mul_task = False
        log("Choose to enable multi-task setting: {}".format(self.to_mul_task))
        return super()._proc_parameter(parameter)
        
    def _gen_task_mask(self):
        self.task_classes = {i:[] for i in range(self.segments)}
    
        for j in range(self.class_num):
            self.task_classes[self.labelmap[j]].append(j)

    def _remove_dup_classes(self):
        for k in self.task_classes.keys():
            self.task_classes[k] = list(set(self.task_classes[k]))

class IncrementalDomainClassificationData(ClassificationTaskData):
    def _assign_elem_id(self, dpid,  dp):
        _, label = dp
        return self.labelmap[label]



class IncrementalDataClassificationData(ClassificationTaskData):
    def _post_process(self):
        print("Into IncrementalClassificationData post process")
        self.comparison = []
        for i in range(1,self.segments):
            self.comparison.append((i-1, i))
            self.data_plan[i].extend(self.data_plan[i-1])
        self._remove_dup_classes()
        if debug and DEBUG:
            print(i,self.data_plan[i][:10])
    
    def _proc_parameter(self, parameter):
        self.do_shuffle = True
        assert_keys_in_dict(["segments"], parameter)
        self.segments = parameter["segments"]
        class_num = utils.config["CLASS_NUM"]
        parameter["task_classes"] = [[list(range(class_num))] for i in range(self.segments)]
        super()._proc_parameter(parameter)

    def _assign_elem_id(self, dpid,  dp):
        return dpid % self.segments

    def _define_order(self, k):
        """ Each Task contains all information from before, but includes new ones. 
        Thus order is defined from current position
        """
        return ("all", None)

class SequentialOrder(ClassificationTaskData):
    def _post_process(self):
        print("Into IncrementalClassificationData post process")
        self.comparison = []
        for i in range(1,self.segments):
            self.comparison.append((i-1, i))
            self.data_plan[i].extend(self.data_plan[i-1])
            #remove the duplicate items
            self.task_classes[i] = self.task_classes[i-1] + self.task_classes[i]
        self._remove_dup_classes()
        if debug and DEBUG:
            print(i,self.data_plan[i][:10])
    def _define_order(self, k):
        """ Each Task contains all information from before, but includes new ones. 
        Thus order is defined from current position
        """
        return ("from", self.order.index(k))

class ConcurrentOrder(ClassificationTaskData):
    def _post_process(self):
        self.combine_key = self.segments
        self.data_plan[self.combine_key] = []
        self.order.append(self.combine_key)
        self.comparison = []
        for i in range(0,self.segments):
            self.comparison.append((i, self.combine_key))
            self.data_plan[self.combine_key].extend(self.data_plan[i])
            self.task_classes[self.combine_key] += self.task_classes[i]
        self._remove_dup_classes()

    def _define_order(self, k):
        """ Each task is concurrently trained.
        Thus we define the order as 
        """
        combine_id = self.order.index(self.combine_key)
        return ("in", [self.order.index(k),combine_id])

IData_CD = IncrementalDataClassificationData
IDomain_CD =IncrementalDomainClassificationData
SeqCD = SequentialOrder
ConCD = ConcurrentOrder

class Seq_IDomain_CD(IDomain_CD, SeqCD):
    @staticmethod
    def __name__():
        return "SeqDomainCD"

class Con_IDomain_CD(IDomain_CD, ConCD):
    @staticmethod
    def __name__():
        return "ConDomainCD"

class Seq_IData_CD(IData_CD, SeqCD):
    @staticmethod
    def __name__():
        return "SeqDataCD"


class Con_IData_CD(IData_CD, ConCD):
    @staticmethod
    def __name__():
        return "ConDataCD"

class MultTaskSeqData(MultiTaskDataTransform):
    def __init__(self, order):
        self. tasks = {} # type: Dict[str, SeqTaskData]
        self.task_names = []
        self.order = order

    def add_task_data(self, taskdata:SeqTaskData, taskname:str):
        assert taskname not in self.task_names
        self.tasks[taskname] = taskdata
        self.task_names.append(taskname)
    
    def list_tasks(self):
        return self.task_names

    def get_task_data(self, taskname:str, order:Any, fold:str):
        assert taskname in self.task_names
        return self.tasks[taskname].get_data(order, fold)

    def fill_evaluator(self, evaluator: EvalBase, prefix=""):
        for name in self.task_names:
            self.tasks[name].fill_evaluator(evaluator, prefix + "{}_".format(name))

    def get_task_compare(self, taskname:str, order:Any):
        assert taskname in self.task_names
        compare_pairs = []         
        for compare_pair in self.tasks[taskname].get_comparison():   
            if compare_pair[-1] == order:
                compare_pairs.append(compare_pair[0])  
        return compare_pairs