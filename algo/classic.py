import torch.nn as nn
import torch as T
import torch.nn.functional as F
from ..utils import get_config_default

l2_loss = nn.MSELoss()

"""Learning without forgetting https://arxiv.org/pdf/1606.09282.pdf
    In page 6, λo is a loss balance weight, set to 1"""
beta = get_config_default("lwf_lambda", 1.0)
def knowledge_distill_loss(full_output, prev_model, x, T=2.):
    #output = model(x, full=True)
    with T.no_grad():
        prev_output = prev_model(x)
        prev_output = F.softmax(prev_output / T, dim=1)
    output = prev_model.process_output(full_output)
    output =  F.log_softmax(prev_output / T, dim=1)
    kd_loss =  T.sum(- prev_output * output, dim=1)
    kd_loss = T.mean(kd_loss) * (T**2)

    return kd_loss * beta

