import copy

from torch import nn
from .utils import get_config, debug

DEBUG = True


class ConvergeImprovement():
    def __init__(self, ratio=1e-3, use_loss=False):
        """ For score, a larger score is better. 
            But for a loss, a smaller loss is better."""
        self.ratio = ratio
        self.max_score = None
        self.avg_growth = 1000
        self.decay_rate = get_config("convergence_decay_rate")
        self.small_better = use_loss

    def save_model(self, model):
        self.best_state = copy.deepcopy(model.state_dict())

    def __call__(self, score, step, model):
        if self.max_score is None:
            self.max_score = score
            self.min_score = score
            self.avg_growth = 1
            self.save_model(model)
        else:
            if self.small_better:
                improve_ratio = (self.min_score - score) / self.max_score 
            else:
                improve_ratio = (score - self.max_score) / self.max_score
            self.avg_growth = self.avg_growth*self.decay_rate + improve_ratio * (1-self.decay_rate)
            if score > self.max_score:
                self.max_score = score
                self.save_model(model)
            if score < self.min_score:
                self.min_score = score
        if debug and DEBUG:
            print("growth in step {} is {}".format(step,self.avg_growth))
        if self.avg_growth < self.ratio:
            return True
        else:
            return False

    def restore_best_model(self, model:nn.Module):
        model.load_state_dict(self.best_state)

class NoImprovement():
    def __init__(self, max_step = 10, use_loss=False):
        self.max_step = max_step
        self.max_score = None
        self.step = 0
        #self.max_step_thresh = get_config("convergence_max_step")
        self.small_better = use_loss
    def __call__(self, score, step, model):
        assert isinstance(model, nn.Module)
        if self.max_score is None:
            self.max_score = score
            self.min_score = score
            self.step = 0
            self.best_model = model
        else:
            if (not self.small_better and score > self.max_score) or\
                (self.small_better and score<self.min_score):
                self.max_score = score
                self.min_score = score
                self.best_model = self.save_model(model)
                self.step = 0
            else:
                self.step += 1
        if debug and DEBUG:
            print("no improvement in steps {}".format(self.step))
        if self.step < self.max_step:
            return False
        else:
            return True
    def save_model(self, model):
        self.best_state = copy.deepcopy(model.state_dict())

    def restore_best_model(self, model:nn.Module):
        model.load_state_dict(self.best_state)