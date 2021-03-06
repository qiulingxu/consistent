import torch as T
import torch.nn as nn
from typing import List, Any, Dict, Tuple
from torch.utils.data import random_split
from torch.utils.data.dataset import ConcatDataset, Subset, Dataset
from abc import ABC, abstractmethod
import math
import copy
import random


from .base import TaskDataTransform, EvalBase, MultiTaskDataTransform
from .utils import assert_keys_in_dict, MAGIC, debug, log, dict_index_range, get_key_default
from . import utils
from . import task
from .utils import get_config
import numpy as np

DEBUG = True

class OverideTransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform 

    def __len__(self):
        return self.dataset.__len__()

    def __repr__(self):
        return str(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        #img, target 
        lst = self.dataset[index]
        img = lst[0]
        if self.transform is not None:
            lst = list(lst)
            img = self.transform(img)
            lst[0] = img
        #if target_transform is not None:
        #    target = target_transform(target)
            
        return lst



class AddOutput(Dataset):
    def __init__(self, dataset, add_func):
        self.dataset = dataset
        self.add_func = add_func

    def __len__(self):
        return self.dataset.__len__()

    def __repr__(self):
        return str(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        lst = self.dataset[index]
        assert isinstance(lst, list)
        lst.append(self.add_func(lst))
        return lst

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

class occulusion(T.nn.Module):
    def __init__(self, tid, total):
        super().__init__()
        assert total in [2,4]
        img_sz = get_config("IMG_SIZE")
        if total == 2:
            self.conf = [(img_sz[1]//2, img_sz[1], 0, img_sz[2]),
                        (0, img_sz[1]//2, 0, img_sz[2]),]
        elif total == 4:
            self.conf = [(img_sz[1]//2, img_sz[1], 0, img_sz[2] // 2),
                        (img_sz[1]//2, img_sz[1], img_sz[2] // 2, img_sz[2]),
                        (0, img_sz[1]//2, 0, img_sz[2]//2),
                        (0, img_sz[1]//2, img_sz[2]//2, img_sz[2]), ]
        self.total = total
        self.tid = tid
        self.conf = self.conf[tid]
    def forward(self, img):
        if len(img.shape) == 3:
            img[:, self.conf[0]:self.conf[1], self.conf[2]:self.conf[3]].fill_(0.)
        elif len(img.shape) ==4:
            img[:, :, self.conf[0]:self.conf[1], self.conf[2]:self.conf[3]].fill_(0.)
        return img
    def __repr__(self):
        return "Occulusion_{}_{}".format(self.total, self.tid)

class SeqTaskData(TaskDataTransform, MultiTaskDataTransform):
    def __init__(self, dataset:Dataset, evaluator:EvalBase, metric:nn.Module, taskname:str = "Task",  **parameter):
        self.dataset = dataset
        self.taskname = taskname
        self.parameter = parameter
        self.evaluator = evaluator
        self.metric = metric
        self.do_shuffle = True
        self.training_transform = get_key_default(parameter, "training_transform", None)
        if self.training_transform:
            print("using training transform {}".format(self.training_transform))
        self.testing_transform = get_key_default(parameter, "testing_transform", None)
        if self.testing_transform:
            print("using testing transform {}".format(self.testing_transform))        
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
        assert order in self.order
        assert fold in ["train", "test", "val"]
        if fold == "train":
            return self.data_plan_train[order]
        elif fold == "test":
            return self.data_plan_test[order]
        elif fold == "val":
            return self.data_plan_val[order]

    def merge_data(self):
        for k in self.order:
            self.data_plan[k] = ConcatDataset([self.data_plan_train[k], self.data_plan_test[k], self.data_plan_val[k]])


    def gen_data_plan(self):
        self.define_datafold()
        if self.do_shuffle:
            self.shuffle_data()
        self.data_plan = {}
        self.data_plan_idx = {}
        for dpid, dp in enumerate(self.dataset):
            idx = self._assign_elem_id(dpid, dp)
            if idx not in self.data_plan_idx:
                self.data_plan_idx[idx] = []
            self.data_plan_idx[idx].append(dpid)
        for idx in self.data_plan_idx.keys():
            self.data_plan[idx] = Subset(self.dataset, self.data_plan_idx[idx])
        self.define_order()
        self.split_data()
        self.fill_evaluator(self.evaluator,"perslice_")
        self._add_index()
        self._post_process()
        self.merge_data()
        #for k in self.data_plan.keys():
        #    self.data_plan[k] = T.stack(self.data_plan[k],dim=0)
    def _add_index(self):
        for idx, k in enumerate(self.order):
            self.data_plan_train[k] = AddOutput(self.data_plan_train[k], add_func=lambda dp: idx )

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
            #if l >= 10:
            self.data_plan_test[k],self.data_plan_val[k],self.data_plan_train[k] \
                = self._split_data(self.data_plan[k])

            if self.training_transform is not None:
                self.data_plan_train[k] = OverideTransformDataset(self.data_plan_train[k], self.training_transform)

            if self.testing_transform is not None:
                self.data_plan_test[k] = OverideTransformDataset(self.data_plan_test[k], self.testing_transform)
                self.data_plan_val[k] = OverideTransformDataset(self.data_plan_val[k], self.testing_transform)
            if get_config("occulusion"):
                oc_transform = occulusion(int(k), len(self.order)) 
                self.data_plan_train[k] = OverideTransformDataset(self.data_plan_train[k], oc_transform)
                self.data_plan_val[k] = OverideTransformDataset(self.data_plan_val[k], oc_transform)
                self.data_plan_test[k] = OverideTransformDataset(self.data_plan_test[k], oc_transform)
    def _split_data(self,data):
        #split_num = copy.deepcopy(self.split_num)
        #for i in range(1, self.slices):
        #    split_num[i] += split_num[i-1]
        assert sum(self.split_num) == self.split_fold
        
        """
        lsts = [[] for i in range(self.slices)]
        for idx, dp in enumerate(data):
            slice = idx%self.split_fold 
            for i in range(self.slices):
                if slice < split_num[i]:
                    lsts[i].append(dp)
                    break
        """
        ldata = len(data)
        tot_num = []
        for psplit in self.split_num[:-1]:
            tot_num.append(int(float(psplit)/ self.split_fold *ldata))
        tot_num.append(ldata- sum(tot_num))
         
        lst_dataset = random_split(data, tot_num, generator=T.Generator().manual_seed(42))

        return lst_dataset

    def fill_evaluator(self, evaluator: EvalBase, prefix=""):
        for k in self.order:
            name = "{}_{}_{}_".format(prefix,self.taskname,str(k))
            if len(self.data_plan_val[k]) >= 2:
                #test, val, train = self._split_data(self.data_plan[k]) 
                test, val, train = self.data_plan_test[k], self.data_plan_val[k], self.data_plan_train[k]
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

    # Multi Task Compatible
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

    def get_metric(self, taskname:str):
        assert taskname == self.taskname
        return self.metric

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
        test, val, train = self.data_plan_test[k], self.data_plan_val[k], self.data_plan_train[k]
        
        #if self.l[k] >= 10:
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
                if "segment_random_seed" in parameter:
                    lst = utils.get_fixed_random_index(num=self.class_num, seed=parameter["segment_random_seed"])
                    def map_to_task(label):
                        nonlocal lst
                        return lst[label] // self.class_per_seg
                else:
                    def map_to_task(label):
                        return label // self.class_per_seg

                self.labelmap = {}
                for i in range(self.class_num):
                    self.labelmap[i] = map_to_task(i)
            else:
                self.labelmap = parameter["labelmap"]
                self.segments = len(set(self.labelmap.values()))
            print("Current label mapping is: {}".format(str(self.labelmap)))
            self._gen_task_mask()
        else:
            self.task_classes = parameter["task_classes"]
            self.segments = len(self.task_classes)
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
        #_, label = dp
        label = dp[1]
        return self.labelmap[label]



class IncrementalDataClassificationData(ClassificationTaskData):


    
    def _proc_parameter(self, parameter):
        self.do_shuffle = True
        assert_keys_in_dict(["segments", "segment_random_seed"], parameter)
        self.segments = parameter["segments"]
        class_num = utils.config["CLASS_NUM"]
        self.order_prob = get_key_default(parameter, "order_prob", None)
        if self.order_prob is not None:
            assert len(self.order_prob) == self.segments
        self.elem_id_gen = utils.get_fixed_random_generator(num= self.segments, seed=parameter["segment_random_seed"], prob= self.order_prob)
        parameter["task_classes"] = {i:list(range(class_num)) for i in range(self.segments)}
        super()._proc_parameter(parameter)

    def _assign_elem_id(self, dpid,  dp):
        #return dpid % self.segments
        return next(self.elem_id_gen)

    def _define_order(self, k):
        """ Each Task contains all information from before, but includes new ones. 
        Thus order is defined from current position
        """
        return ("all", None)

class SequentialOrder(ClassificationTaskData):
    def _post_process(self):
        print("Into IncrementalDataClassificationData post process")
        self.comparison = []
        
        def merge_data_plan(data_plan):
            new_data_plan = {0:data_plan[0]}
            for i in range(1,self.segments):
                new_data_plan[i] = ConcatDataset(dict_index_range(data_plan, 0, i+1))
                if debug and DEBUG:
                    print(i,len(new_data_plan[i]))
            return new_data_plan

        for i in range(1,self.segments):
            self.comparison.append((i-1, i))
            self.task_classes[i] += self.task_classes[i-1]
            #self.data_plan[i].extend(self.data_plan[i-1])

        self.data_plan_test = merge_data_plan(self.data_plan_test)
        self.data_plan_val = merge_data_plan(self.data_plan_val)
        self.data_plan_train = merge_data_plan(self.data_plan_train)
        
        self._remove_dup_classes()

    def _define_order(self, k):
        """ Each Task contains all information from before, but includes new ones. 
        Thus order is defined from current position
        """
        return ("from", self.order.index(k))

class ConcurrentOrder(ClassificationTaskData):
    def _post_process(self):
        print("Into ConcurrentOrder post process")
        self.combine_key = self.segments
        self.data_plan[self.combine_key] = []
        self.order.append(self.combine_key)
        def merge_data_plan(data_plan):
            new_data_plan = data_plan
            new_data_plan[self.combine_key] = ConcatDataset(dict_index_range(self.data_plan, 0, self.segments+1))
            if debug and DEBUG:
                print(i,len(new_data_plan[self.combine_key]))
            return new_data_plan        
        self.comparison = []
        for i in range(0,self.segments):
            self.comparison.append((i, self.combine_key))
            self.task_classes[self.combine_key] += self.task_classes[i]

        self.data_plan_test = merge_data_plan(self.data_plan_test)
        self.data_plan_val = merge_data_plan(self.data_plan_val)
        self.data_plan_train = merge_data_plan(self.data_plan_train)
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

    def get_metric(self, taskname:str):
        assert taskname in self.task_names
        return self.tasks[taskname].metric

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



