from abc import ABC, abstractmethod
from typing import Dict, Any
from math import inf
import torch as T
import torch.nn as nn
import copy
from . import utils
from .base import Task,TaskDataTransform,MultiTaskDataTransform, EvalBase
from .utils import log, device, get_key_default, debug, freeze, PytorchModeWrap as PMW
from .net import ClassificationMask
from .taskdata import ClassificationTaskData
from .utils import get_config
from .metric import eval_score

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



class ConvergeImprovement():
    def __init__(self, ratio=1e-3):
        self.ratio = ratio
        self.max_score = None
        self.avg_growth = 1000
        self.decay_rate = get_config("convergence_decay_rate")

    def save_model(self, model):
        self.best_state = copy.deepcopy(model.state_dict())

    def __call__(self, score, step, model):
        if self.max_score is None:
            self.max_score = score
            self.avg_growth = 1
            self.save_model(model)
        else:
            improve_ratio = (score - self.max_score) / self.max_score
            self.avg_growth = self.avg_growth*self.decay_rate + improve_ratio * (1-self.decay_rate)
            if score > self.max_score:
                self.max_score = score
                self.save_model(model)
        if debug and DEBUG:
            print("growth in step {} is {}".format(step,self.avg_growth))
        if self.avg_growth < self.ratio:
            return True
        else:
            return False

    def restore_best_model(self, model:nn.Module):
        model.load_state_dict(self.best_state)
class NoImprovement():
    def __init__(self, max_step = 10):
        self.max_step = max_step
        self.max_score = None
        self.step = 0
        self.max_step_thresh = get_config("convergence_max_step")

    def __call__(self, score, step, model):
        assert isinstance(model, nn.Module)
        if self.max_score is None:
            self.max_score = score
            self.step = 0
            self.best_model = model
        else:
            if score > self.max_score:
                self.max_score = score
                self.best_model = copy.deepcopy(model)
                self.step = 0
            else:
                self.step += 1
        if debug and DEBUG:
            print("no improvement in steps {}".format(self.step))
        if self.step < self.max_step:
            return False
        else:
            return True

    def get_best_model(self):
        return self.best_model

class VanillaTrain(Task):
    def __init__(self,  granularity, evalulator:EvalBase, taskdata:MultiTaskDataTransform, task_prefix, **parameter:dict):
        
        assert granularity in ["epoch", "batch", "converge"]
        self.granularity = granularity
        self.task_prefix = task_prefix
        self.taskdata  = taskdata
        self.multi_task_flag = True
        
        self.evaluator = evalulator
        self.ipv_threshold = get_config("convergence_improvement_threshold")
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
                    self.converge[tn] = ConvergeImprovement(self.ipv_threshold)
            
                while True:
                    if self.multi_task_flag == False:
                        self.curr_task_name = list(task2model.keys())[0]
                        ctn = self.curr_task_name
                        self.curr_model[ctn] = self.model_process(ctn, \
                            self.curr_model[ctn], order, step)
                        train_loop(self.curr_model, self.curr_train_data_loader)
                        sc = self.perf_metric[ctn](self.curr_model[ctn], self.curr_val_data_loader[ctn])                
                        if self.converge[ctn](sc, step, self.curr_model[ctn]):
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
            self._train_single(task2model[task_name], \
                dataset[task_name], \
                prev_models[task_name], 
                device=self.device, **karg)
        else:
            assert False, "Please implement it"
    
    def eval(self, model, dataset, prev_models, **karg):
        self._eval(model, dataset, prev_models, device=self.device, **karg)

    def _train_single(self, model, dataset, prev_models,  **karg):
        """ This function can be used for training single task with multiple stage. It bypass the multutask feature"""
        print("Please implement train single task function") 
    
    def _train(self, task2model, dataset, prev_models,  **karg):
        print("Please implement train multi task function") 

    def _model_process(self, task_name, model: nn.Module, key, step):
        return model

    @abstractmethod
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
    