import torch.nn as nn
import torch as T
from ..utils import get_config_default

l2_loss = nn.MSELoss()

"""Learning without forgetting https://arxiv.org/pdf/1606.09282.pdf
    In page 6, λo is a loss balance weight, set to 1"""
beta = get_config_default("lwf_lambda", 1.0)
def knowledge_distill_loss(model, prev_model, x):
    output = model(x, full=True)
    prev_output = prev_model(x)
    output = prev_model.process_output(output)
    return l2_loss(output, prev_output) * beta

