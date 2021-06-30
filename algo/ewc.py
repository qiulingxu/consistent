# https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
# No Liceense indicated
from copy import deepcopy
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader,RandomSampler

from ..utils import PytorchModeWrap as PMW, PytorchFixWrap as PFW
lam = 10.0
class EWC(nn.Module):
    def __init__(self, dataset:Iterable, to_data_loader, max_data=None):
        super().__init__()
        self.dataset = dataset
        self.ld = min(1000, len(dataset))
        sampler = RandomSampler(range(self.ld),replacement=True, num_samples=self.ld)
        self.dataloader = to_data_loader(dataset, batch_size=1, sampler=sampler, shuffle=False)
        self.softmax = nn.CrossEntropyLoss()

    def set_model(self, model:nn.Module, var_lst):
        self.model = model
        #print(model.named_parameters)"
        self.var_lst = var_lst #[v for v in var_lst if v.find("linear")==-1 ]
        self.params = {n: p for n, p in model.named_parameters() if n in var_lst}
        #print("param",var_lst)

    def eval_fisher(self):
        #can only run offline
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        
        with PMW(self.model,False):
            with PFW(self.model,self.var_lst,True): 
                for idx, (input, label) in enumerate(self.dataloader):
                    self.model.zero_grad()
                    input = variable(input)
                    label = label.to(input.device)
                    output = self.model(input).view(1, -1)
                    label = self.model.process_labels(label)
                    #loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                    loss = self.softmax(output, label)
                    #print(loss)
                    assert not torch.isnan(loss).any(), "NAN" + str(loss)
                    loss.backward()
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if n in self.var_lst:
                                precision_matrices[n] += p.grad ** 2 / self.ld
                                assert not torch.isnan(precision_matrices[n]).any(), "NAN" + n
                        #print(precision_matrices)
                    #i = input()
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self._precision_matrices = precision_matrices

        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def penalty(self, model: nn.Module):
        loss = 0
        #print(model.named_parameters)
        for n, p in model.named_parameters():
            if n in self.var_lst:
                _loss = self._precision_matrices[n] * ((p - self._means[n]) ** 2)
                """if torch.isnan(_loss).any():
                    print("NAN" + n)
                    print(p, self._means[n])
                    print(self._precision_matrices[n]) 
                    assert False"""
                loss += _loss.sum()
        return loss * lam

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return t#Variable(t, **kwargs)




class EWC_ORIGIN(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
    

def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct / len(data_loader.dataset)