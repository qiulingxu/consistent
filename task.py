import torch as T

from .base import Task,TaskDataTransform, EvalBase
from .utils import log, device

def classifier_perf_metric(model, test_data):
    model = model.to(device)
    model.eval()
    tot_corr = 0
    cnt = 0
    for dp in test_data:
        x, y = dp
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        y_pred = T.argmax(output, dim=1)
        corr = T.eq(y_pred, y).float()
        tot_corr += T.sum(corr)
        cnt += list(corr.shape)[0]
    return tot_corr.numpy()/cnt

class VanillaTrain(Task):
    def __init__(self, parameter:dict, granularity, evalulator:EvalBase, taskdata:TaskDataTransform, task_prefix):
        
        assert granularity in ["epoch", "batch", "converge"]
        self.granularity = granularity
        self.task_prefix = task_prefix
        self.taskdata  = taskdata
        self.process_parameter(parameter)

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

    
    def controlled_train(self, model):

        def train_loop(model, traindata,):
            nonlocal step, tot_step
            self.train(model, traindata)
            step += 1
            tot_step += 1

        self.prev_models = []
        tot_step = 0
        for k in self.taskdata.order:
            curr_train_data = self.taskdata.data_plan_train[k]
            curr_test_data = self.taskdata.data_plan_test[k]
            curr_train_data = self.process_data(curr_train_data)
            curr_test_data = self.process_data(curr_test_data)
            step = 0
            if self.granularity == "converge":
                while True:
                    train_loop(model, curr_train_data)
                    sc = self.perf_metric(model, curr_test_data)                
                    if self.converge(sc, step):
                        log("Task {} converges after {} steps".format(k, step))
                        break
            else:
                assert False, "Implement other time slice definition"

    @abstractmethod           
    def converge(self, criterion):
        return True
    
    @abstractmethod
    def train(self, model, dataset, prev_models):
        pass

