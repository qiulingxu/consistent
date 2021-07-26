import torch as T
import torch.nn as nn
import numpy as np
from typing import List, Type
from .base import ClassificationModule
from .utils import config

def error(s):
    assert False, s
    return None

def ClassificationMask(cls):
    class wrap_cls(cls, ClassificationModule):
        def __init__(self, *arg, **karg):
            super().__init__(*arg, **karg)
            self.sublabels(list(range(config["CLASS_NUM"])))

        def get_linear(self):
            return self.linear

        def sublabels(self, labels):
            self.labels = T.tensor(labels, dtype=T.int64)
            self.map= {l : idx for idx, l in enumerate(labels)}

        def process_labels(self, labels):
            def process(labels):
                labels= [self.map[l] if l in self.map else error("Invalid label {} in current task".format(l)) for l in labels]
                return labels
            if T.is_tensor(labels):
                device = labels.device
                _labels = labels.cpu().numpy()
                _labels = process(_labels)
                labels = T.tensor(_labels, dtype = labels.dtype).to(device)
            elif isinstance(labels, list):
                labels = process(labels)
            return labels

        def reset_head(self):
            assert False
            with T.no_grad():
                nn.init.xavier_uniform_(self.get_linear().weight)

        def process_output(self, output):
            out = T.index_select(output, dim=1, index=self.labels.to(output.device))
            assert list(out.size())[1] == len(self.labels)
            return out

        def forward(self, x, full=False):
            ret = super().forward(x)
            if full:
                return ret
            else:
                return self.process_output(ret)
    return wrap_cls


class AvgNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_list = nn.ModuleList()
        self.model_num = 0
        self.weights = []
    def add_net(self, model:ClassificationModule, weight=1.0):
        self.model_num += 1
        self.model_list.append(model)
        self.weights.append(weight)
    def forward(self, x, full=False):
        o = 0 
        tot = 0
        for i in range(self.model_num):
            o += self.model_list[i](x, full=True) * self.weights[i]
            tot += self.weights[i]
        o = o / tot
        if full:
            return o
        else:
            return self.model_list[-1].process_output(o)

    def process_labels(self, labels):
        return self.model_list[-1].process_labels(labels)
if __name__ == "__main__":
    class naive_nn(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            lst = list(range(10))
            lst = np.stack([lst]*10, axis=0)
            return T.tensor(lst)
    net = ClassificationMask(naive_nn)()
    net.sublabels([1,5,9])
    print("process labels", net.process_labels([1,1,5,5,9,9]))
    print(net(T.zeros(0)))