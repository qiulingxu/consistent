from abc import ABC, abstractmethod
from typing import Dict
from math import inf
import torch as T
import torch.nn as nn
import copy
from . import utils
from .base import Task,TaskDataTransform,MultiTaskDataTransform, EvalBase
from .utils import log, device, get_key_default, debug, PytorchModeWrap as PMW
from .net import ClassificationMask
from .taskdata import ClassificationTaskData
from .utils import get_config

DEBUG = True

def classifier_perf_metric(model, test_data):
    model = model.to(device)
    with PMW(model, training = False):
        tot_corr = 0
        cnt = 0
        for dp in test_data:
            x, y = dp
            y = model.process_labels(y)
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            y_pred = T.argmax(output, dim=1)
            corr = T.eq(y_pred, y).float()
            tot_corr += T.sum(corr)
            cnt += list(corr.shape)[0]
    return tot_corr.cpu().numpy()/cnt

class ConvergeImprovement():
    def __init__(self, ratio=1e-3):
        self.ratio = ratio
        self.max_score = None
        self.avg_growth = 1000
        self.decay_rate = get_config("convergence_decay_rate")

    def __call__(self, score, step):
        if self.max_score is None:
            self.max_score = score
            self.avg_growth = 1
        else:
            improve_ratio = (score - self.max_score) / self.max_score
            self.avg_growth = self.avg_growth*self.decay_rate + improve_ratio * (1-self.decay_rate)
            self.max_score = max(score, self.max_score)
        if debug and DEBUG:
            print("growth in step {} is {}".format(step,self.avg_growth))
        if self.avg_growth < self.ratio:
            return True
        else:
            return False

class VanillaTrain(Task):
    def __init__(self,  granularity, evalulator:EvalBase, taskdata:MultiTaskDataTransform, task_prefix, **parameter:dict):
        
        assert granularity in ["epoch", "batch", "converge"]
        self.granularity = granularity
        self.task_prefix = task_prefix
        self.taskdata  = taskdata
        self.multi_task_flag = True
        self.process_parameter(parameter)
        
        self.evaluator = evalulator
        self.ipv_threshold = get_config("convergence_improvement_threshold")
        self.iscopy = get_key_default(parameter, "iscopy", True, type=bool)
        self.device = get_key_default(parameter, "device", device, type=str)
        self.max_epoch = 1e9
    def process_parameter(self, parameter):
        if self.granularity == "converge":
            if "max_epoch" in parameter:
                self.max_epoch = parameter["max_epoch"]
            if "perf_metric" in parameter:
                self.perf_metric = parameter["perf_metric"]
            else:
                self.perf_metric = classifier_perf_metric
        elif self.granularity == "epoch":
            self.epoch = parameter["epoch"]
        else:
            assert False

    def get_name(self,):
        return "{}_vanilla_epoch{}_gran{}" .format(self.task_prefix, self.epoch, self.granularity)

    @abstractmethod
    def process_data(self, dataset, mode="train"):
        assert mode in ["train", "test"]
        return dataset

    def controlled_train_single_task(self, model, **karg):
        self.multi_task_flag = False
        self.controlled_train(self, {"Task":model}, **karg)
        self.multi_task_flag = True

    def controlled_train(self, task2model: Dict[str, nn.Module], **karg):

        def train_loop(model, traindata,):
            nonlocal step, tot_step
            if self.granularity == "converge":
                karg["epoch"] = step
            with PMW(model, training=True):
                self.train(model = model, \
                        dataset=traindata, \
                        prev_models=self.prev_models,\
                        **karg)
            step += 1
            tot_step += 1

        self.prev_models = {}
        tot_step = 0
        self.pre_train()
        for order in self.taskdata.order:
            # These variables are for current time slice
            self.curr_order = order
            self.curr_train_data = {}
            self.curr_test_data = {}
            self.curr_val_data = {}
            self.curr_train_data_loader = {}
            self.curr_test_data_loader = {}
            self.curr_val_data_loader = {}
            self.compare_pairs = {}
            self.curr_model = {}
            for task_name, model in task2model.items():
                ### Process the data

                self.curr_train_data[task_name] = self.taskdata.get_task_data(task_name, order, "train")
                self.curr_test_data[task_name] = self.taskdata.get_task_data(task_name, order, "test")
                self.curr_val_data[task_name] = self.taskdata.get_task_data(task_name, order, "val")
                if debug and DEBUG:
                    print("Data num for train {}, test {} and val {}".\
                        format(len(self.curr_train_data), len(self.curr_test_data), len(self.curr_val_data)))
                self.curr_train_data_loader[task_name] = self.process_data(self.curr_train_data, mode="train")
                self.curr_test_data_loader[task_name] = self.process_data(self.curr_test_data, mode="eval")
                self.curr_val_data_loader[task_name] = self.process_data(self.curr_val_data, mode ="eval")
            
                ### Process the depencency edge
                self.compare_pairs[task_name] = self.taskdata.get_task_compare(task_name, order)
                self.curr_model[task_name] = self.model_process(task_name, model, order, step)
            step = 0
            if self.granularity == "converge":
                self.converge = ConvergeImprovement(self.ipv_threshold)
                while True:
                    
                    train_loop(model, self.curr_train_data_loader)
                    sc = self.perf_metric(model, self.curr_val_data_loader)                
                    if self.converge(sc, step):
                        log("Task {} converges after {} steps".format(order, step))
                        break
                    if step > self.max_epoch:
                        log("Task {} reaches max epochs after {} steps".format(order, step))
                        break
            else:
                assert False, "Implement other time slice definition"
            for task_name, model in task2model.items():
                self.curr_model[task_name] = self.model_process(task_name, model, order, -1)
            self.evaluator.eval(self.curr_model)    
            log("Measure",self.evaluator.measure())
            if self.iscopy:
                self.curr_model = self.copy(self.curr_model)
                self.prev_models[order] = self.curr_model
            else:
                self.prev_models = None
                self.curr_model = self.curr_model
            self.post_task()
            self.eval(model=model,
                        dataset=self.curr_test_data_loader, \
                        prev_models=self.prev_models,\
                        **karg)
    def copy(self, models):
        return copy.deepcopy(models)
    #def converge(self, criterion):
    #    return True
    def post_task(self):
        pass

    def pre_train(self):
        pass

    def model_process(self, task_name:str, model: nn.Module, key:str, step:int):
        """step = -1 when it finishes training"""
        ret = self._model_process(task_name, model, key, step)
        assert ret is not None, "Please implement the processd model after iteration"
        return ret
    
    def train(self, model, dataset, prev_models, **karg):
        self._train(model, dataset, prev_models, device=self.device, **karg)
    
    def eval(self, model, dataset, prev_models, **karg):
        self._eval(model, dataset, prev_models, device=self.device, **karg)
    
    @abstractmethod
    def _train(self, model, dataset, prev_models,  **karg):
        pass

    def _model_process(self, task_name, model: nn.Module, key, step):
        return model

    @abstractmethod
    def _eval(self, model, dataset, prev_models,  **karg):
        pass

class ClassificationTrain(VanillaTrain):
    def __init__(self, granularity, evalulator:EvalBase, taskdata:ClassificationTaskData, task_prefix, **parameter:dict):
        super().__init__(granularity,evalulator,taskdata, task_prefix, **parameter)

    def model_process(self, model: nn.Module, key:str, step:int): # type: ignore[override]
        if step == 0:
            tp = get_config("classification_model_process")
            if tp.find("mask") >= 0:
                model.sublabels(self.taskdata.task_classes[key])
            if tp.find("reset") >= 0:
                model.get_linear().reset_parameters()
        return self._model_process(model, key, step)
    