import torch as T

from .base import Task



class VanillaTrain(Task):
    def __init__(self, parameter:dict, granularity, evalulator, task_prefix):
        
        assert granularity in ["epoch", "batch"]
        self.granularity = granularity
        self.task_prefix = task_prefix
        self.process_parameter(parameter)

    def process_parameter(self, parameter):
        self.epoch = parameter["epoch"]

    def get_name(self,):
        return "{}_vanilla_epoch{}_gran{}" .format(self.task_prefix, self.epoch, self.granularity)

    @abstractmethod
    def process_data(self, dataset):
        return dataset

    @abstractmethod
    def controlled_train(self, Model, dataset):
        pass