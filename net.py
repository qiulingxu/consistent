import torch as T
import torch.nn as nn

class ClassificationMask(nn.Module):
    def __init__(self, net):
        super().__init__()
    
    def sublabels(self, labels):
        self.labels = T.tensor(labels, dtype=T.int32)
        self.map= {l : idx for idx, l in enumerate(labels)}

    def process_labels(self, labels):
        labels = [self.map[l] for l in labels]
        return labels

    def forward(self, x):
        ret = super().forward(x)
        T.index_select(ret, dim=1, index=)