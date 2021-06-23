import torch as T
import torch.nn as nn
import numpy as np
from typing import List, Type
from .utils import config

def ClassificationMask(cls):
    class wrap_cls(cls):
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
                labels= [self.map[l] if l in self.map else -1 for l in labels]
                return labels
            if T.is_tensor(labels):
                device = labels.device
                _labels = labels.cpu().numpy()
                _labels = process(_labels)
                labels = T.tensor(_labels, dtype = labels.dtype).to(device)
            elif isinstance(labels, list):
                labels = process(labels)
            return labels

        def process_output(self, output):
            return T.index_select(output, dim=1, index=self.labels.to(output.device))

        def forward(self, x, full=False):
            ret = super().forward(x)
            if full:
                return ret
            else:
                return self.process_output(ret)
    return wrap_cls



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