from abc import ABC, abstractmethod
from typing import Dict, Any
from math import inf
import torch as T
import torch.nn as nn
import copy
from . import utils
from .base import Task,TaskDataTransform,MultiTaskDataTransform, EvalBase
from .utils import get_config_default, log, device, get_key_default, debug, freeze, PytorchModeWrap as PMW
from .net import ClassificationMask
from .taskdata import ClassificationTaskData
from .utils import get_config
from .metric import eval_score
from .converge import NoImprovement, ConvergeImprovement
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
            cnt += y.size(0)
    return (tot_corr/cnt).cpu().numpy()




class VanillaTrain(Task):
    def __init__(self,  granularity, evalulator:EvalBase, taskdata:MultiTaskDataTransform, task_prefix, **parameter:dict):
        
        assert granularity in ["epoch", "batch", "converge"]
        self.granularity = granularity
        self.task_prefix = task_prefix
        self.taskdata  = taskdata
        self.multi_task_flag = True
        
        self.evaluator = evalulator
        if granularity == "converge":
            # Two ways to definee  conveergence
            converge_def = get_config("convergence_method")
            if converge_def == "max_step":
                self.ipv_max_step = get_config_default("convergence_improvement_max_step", False)
                self.ipv_threshold = False
            else:
                self.ipv_max_step = False
                self.ipv_threshold = get_config_default("convergence_improvement_threshold", False)
            
            assert not self.ipv_threshold or not self.ipv_max_step
        self.iscopy = get_key_default(parameter, "iscopy", True, type=bool)
        self.device = get_key_default(parameter, "device", device, type=str)
        self.max_epoch = 1e9

        self.perf_metric = {}
        self.task_var = {}
        self.process_parameter(parameter)

    def process_parameter(self, parameter):
        if self.granularity == "converge":
            if "max_epoch" in parameter:
                self.max_epoch = parameter["max_epoch"]
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
        self.controlled_train({"Task":model}, **karg)
        self.multi_task_flag = True

    def controlled_train(self, task2model: Dict[str, nn.Module], **karg):

        def train_loop(task2model, traindata,):
            nonlocal step, tot_step
            if self.granularity == "converge":
                karg["epoch"] = step
            with PMW(task2model, training=True):
                self.train(task2model = task2model, \
                        dataset=traindata, \
                        prev_models=self.prev_models,\
                        **karg)
            step += 1
            tot_step += 1

        self.prev_models = {} # type: Dict[str, Dict[Any, nn.Module]]
        tot_step = 0
        self.tasks = task2model.keys()
        self.pre_train()
        for tn in self.tasks:
            self.prev_models[tn] = {}
        self.curr_model = task2model
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
            #self.curr_model = {}

            for task_name in self.tasks:
                ### Process the data
                self.perf_metric[task_name] = eval_score(self.taskdata.get_metric(task_name))
                self.curr_train_data[task_name] = self.taskdata.get_task_data(task_name, order, "train")
                self.curr_test_data[task_name] = self.taskdata.get_task_data(task_name, order, "test")
                self.curr_val_data[task_name] = self.taskdata.get_task_data(task_name, order, "val")
                if debug:
                    print(order)
                    print(len(self.curr_train_data[task_name]))
                if debug and DEBUG:
                    print("Data num for train {}, test {} and val {}".\
                        format(len(self.curr_train_data[task_name]), len(self.curr_test_data[task_name]), len(self.curr_val_data[task_name])))
                self.curr_train_data_loader[task_name] = self.process_data(self.curr_train_data[task_name], mode="train")
                self.curr_test_data_loader[task_name] = self.process_data(self.curr_test_data[task_name], mode="eval")
                self.curr_val_data_loader[task_name] = self.process_data(self.curr_val_data[task_name], mode ="eval")
            
                ### Process the depencency edge
                self.compare_pairs[task_name] = self.taskdata.get_task_compare(task_name, order)
            step = 0
            if self.granularity == "converge":
                self.converge = {}
                for tn in self.tasks:
                    if self.ipv_threshold:
                        self.converge[tn] = ConvergeImprovement(self.ipv_threshold)
                    if self.ipv_max_step:
                        self.converge[tn] = NoImprovement(self.ipv_max_step, use_loss=True)
            
                while True:
                    if self.multi_task_flag == False:
                        self.curr_task_name = list(task2model.keys())[0]
                        ctn = self.curr_task_name
                        self.curr_model[ctn] = self.model_process(ctn, \
                            self.curr_model[ctn], order, step)
                        train_loop(self.curr_model, self.curr_train_data_loader)
                        if self.ipv_threshold:
                            sc = self.perf_metric[ctn](self.curr_model[ctn], self.curr_val_data_loader[ctn])                
                            if self.converge[ctn](sc, step, self.curr_model[ctn]):
                                log("Task {} converges after {} steps".format(order, step))
                                break
                        if self.ipv_max_step:
                            loss = self.eval(self.curr_model, self.curr_train_data_loader,
                                    prev_models=self.prev_models, **karg)
                            if self.converge[ctn](loss, step, self.curr_model[ctn]):
                                log("Task {} converges after {} steps".format(order, step))
                                break
                        if step > self.max_epoch:
                            log("Task {} reaches max epochs after {} steps".format(order, step))
                            break
                    else:
                        assert False, "not Implemented"
                for tn in self.tasks:
                    self.converge[tn].restore_best_model(self.curr_model[tn])
            else:
                assert False, "Implement other time slice definition"
            for task_name, model in self.curr_model.items():
                self.curr_model[task_name] = self.model_process(task_name, model, order, -1)
            self.evaluator.eval(self.curr_model)    
            log("Measure",self.evaluator.measure())
            if self.iscopy:
                self.last_model = self.copy(self.curr_model)
            else:
                assert False
                self.last_model = self.curr_model
            for tn in self.tasks:
                var_lst = freeze(self.last_model[tn])
                self.last_model[tn].eval()
                if tn in self.task_var:
                    assert self.task_var[tn] == var_lst, "Task parameters are changed for {}".format(tn)
                else:
                    self.task_var[tn] = var_lst
                self.prev_models[tn][order] = self.last_model[tn]
            self.post_task()
            """self.eval(model=model,
                        dataset=self.curr_test_data_loader, \
                        prev_models=self.prev_models,\
                        **karg)"""
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
    

    def train(self, task2model, dataset, prev_models, **karg):
        if self.multi_task_flag == False:
            # Only one task is used here
            assert len(task2model) == 1
            task_name = list(task2model.keys())[0]
            return self._train_single(task2model[task_name], \
                dataset[task_name], \
                prev_models[task_name], 
                device=self.device, **karg)
        else:
            assert False, "Please implement it"
    
    def eval(self, task2model, dataset, prev_models, **karg):
        if self.multi_task_flag == False:
            # Only one task is used here
            assert len(task2model) == 1
            task_name = list(task2model.keys())[0]
            return self._eval_single(task2model[task_name], \
                dataset[task_name], \
                prev_models[task_name], 
                device=self.device, **karg)
        else:
            assert False, "Please implement it"
            self._eval(model, dataset, prev_models, device=self.device, **karg)

    def _train_single(self, model, dataset, prev_models,  **karg):
        """ This function can be used for training single task with multiple stage. It bypass the multutask feature"""
        print("Please implement train single task function") 

    def _eval_single(self, model, dataset, prev_models,  **karg):
        print("Please implement eval single task function") 

    def _train(self, task2model, dataset, prev_models,  **karg):
        print("Please implement train multi task function") 

    def _model_process(self, task_name, model: nn.Module, key, step):
        return model

    def _eval(self, model, dataset, prev_models,  **karg):
        pass

class ClassificationTrain(VanillaTrain):
    def __init__(self, granularity, evalulator:EvalBase, taskdata:ClassificationTaskData, task_prefix, **parameter:dict):
        super().__init__(granularity,evalulator,taskdata, task_prefix, **parameter)

    def model_process(self, task_name:str, model: nn.Module, key:str, step:int): # type: ignore[override]
        if step == 0:
            tp = get_config("classification_model_process")
            if tp.find("mask") >= 0:
                model.sublabels(self.taskdata.tasks[task_name].task_classes[key])
            if tp.find("reset") >= 0:
                model.get_linear().reset_parameters()
        return super().model_process(task_name, model, key, step)
    