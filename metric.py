import torch as T
import torch.nn as nn

class MetricClassification(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: T.Tensor, data: dict):
        x, y = data["x"], data["y"]
        assert T.is_tensor(y)
        y_pred = T.argmax(output, dim=1)
        return T.eq(y_pred, y).float()