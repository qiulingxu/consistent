from abc import ABC, abstractmethod
from math import inf
import torch as T
import torch.nn as nn
import copy

from .base import Task,TaskDataTransform, EvalBase
from .utils import log, device, get_key_default, debug, PytorchModeWrap as PMW
from .net import ClassificationMask
from .taskdata import ClassificationTaskData

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
        self.decay_rate = 0.7

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
    def __init__(self, parameter:dict, granularity, evalulator:EvalBase, taskdata:TaskDataTransform, task_prefix):
        
        assert granularity in ["epoch", "batch", "converge"]
        self.granularity = granularity
        self.task_prefix = task_prefix
        self.taskdata  = taskdata
        self.process_parameter(parameter)
        
        self.evaluator = evalulator
        self.ipv_threshold = get_key_default(parameter, "ipv_threshold", 1e-3, type=float)
        self.iscopy = get_key_default(parameter, "iscopy", True, type=bool)
        self.device = get_key_default(parameter, "device", device, type=str)
    def process_parameter(self, parameter):
        if self.granularity == "converge":
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

    
    def controlled_train(self, model, **karg):

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

        self.prev_models = []
        tot_step = 0
        for k in self.taskdata.order:
            curr_train_data = self.taskdata.data_plan_train[k]
            curr_test_data = self.taskdata.data_plan_test[k]
            curr_val_data = self.taskdata.data_plan_val[k]
            if debug and DEBUG:
                print("Data num for train {}, test {} and val {}".\
                    format(len(curr_train_data), len(curr_test_data), len(curr_val_data)))
            curr_train_data_loader = self.process_data(curr_train_data, mode="train")
            curr_test_data_loader = self.process_data(curr_test_data, mode="eval")
            curr_val_data_loader = self.process_data(curr_val_data, mode ="eval")
            step = 0
            if self.granularity == "converge":
                self.converge = ConvergeImprovement(self.ipv_threshold)
                while True:
                    model = self.model_process(model, k, step)
                    train_loop(model, curr_train_data_loader)
                    sc = self.perf_metric(model, curr_val_data_loader)                
                    if self.converge(sc, step):
                        log("Task {} converges after {} steps".format(k, step))
                        break
            else:
                assert False, "Implement other time slice definition"
            self.evaluator.eval(model)    
            print("Measure",self.evaluator.measure())
            if self.iscopy:
                curr_model = copy.deepcopy(model)
                self.prev_models = curr_model
            else:
                self.prev_models.append(model)
            self.eval(model=model,
                        dataset=curr_test_data_loader, \
                        prev_models=self.prev_models,\
                        **karg)
    #def converge(self, criterion):
    #    return True

    def model_process(self, model: nn.Module, key:str, step:int):
        return self._model_process(model, key, step)
    
    def train(self, model, dataset, prev_models, **karg):
        self._train(model, dataset, prev_models, device=self.device, **karg)
    
    def eval(self, model, dataset, prev_models, **karg):
        self._eval(model, dataset, prev_models, device=self.device, **karg)
    
    @abstractmethod
    def _train(self, model, dataset, prev_models,  **karg):
        pass

    def _model_process(self, model: nn.Module, key, step):
        return model

    @abstractmethod
    def _eval(self, model, dataset, prev_models,  **karg):
        pass

class ClassificationTrain(VanillaTrain):
    def __init__(self, parameter:dict, granularity, evalulator:EvalBase, taskdata:ClassificationTaskData, task_prefix):
        super().__init__(parameter, granularity,evalulator,taskdata, task_prefix)

    def model_process(self, model: nn.Module, key:str, step:int): # type: ignore[override]
        if step == 0:
            model.sublabels(self.taskdata.task_classes[key])
        return self._model_process(model, key, step)
    