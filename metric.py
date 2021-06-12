import torch as T
import torch.nn as nn
from .net import ClassificationMask

class MetricClassification(nn.Module):
    def __init__(self, model:ClassificationMask):
        super().__init__()
        self.model = model

    def forward(self, output: T.Tensor, data: dict):
        x, y = data["x"], data["y"]
        assert T.is_tensor(y)
        y = self.model.process_labels(y)
        y_pred = T.argmax(output, dim=1)
        return T.eq(y_pred, y).float()