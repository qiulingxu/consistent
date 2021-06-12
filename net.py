import torch as T
import torch.nn as nn
import numpy as np

class ClassificationMask(nn.Module):
    def __init__(self, net):
        super().__init__()
    
    def sublabels(self, labels):
        self.labels = T.tensor(labels, dtype=T.int32)
        self.map= {l : idx for idx, l in enumerate(labels)}

    def process_labels(self, labels):
        if T.is_tensor(labels):
            _labels = labels.cpu().numpy()
            _labels = [self.map[l] for l in _labels]
            labels = T.tensor(_labels, dtype = labels.dtype)
        elif isinstance(labels, list):
            labels = [self.map[l] for l in labels]
        return labels

    def forward(self, x):
        ret = super().forward(x)
        return T.index_select(ret, dim=1, index=self.labels)