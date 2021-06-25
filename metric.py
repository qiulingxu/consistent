import torch as T
import torch.nn as nn
from .base import ClassificationModule
from .utils import device, PytorchModeWrap as PMW
class MetricClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_task = "classifier"

    def forward(self, output: T.Tensor, data: dict, model:ClassificationModule):
        x, y = data["x"], data["y"]
        assert T.is_tensor(y)
        y = model.process_labels(y)
        y_pred = T.argmax(output, dim=1)
        return T.eq(y_pred, y).float()

class eval_score(nn.Module):
    def __init__(self, metric:nn.Module):
        super().__init__()
        self.metric = metric
        assert self.metric.base_task in ["classifier"], "Please add others data processing"
    def forward(self, model:ClassificationModule, test_data):
        model = model.to(device)
        with PMW(model, training = False):
            tot_sc = 0
            cnt = 0
            
            for dp in test_data:
                x, y = dp
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                sc = self.metric(output, {"x":x, "y":y}, model)
                tot_sc += T.sum(sc)
                cnt += list(sc.shape)[0]
        return tot_sc.cpu().numpy()/cnt

classifier_correct = eval_score(MetricClassification())