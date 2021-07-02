import torch.nn as nn
import torch as T
import torch.nn.functional as F
from ..utils import get_config

l2_loss = nn.MSELoss()

"""Learning without forgetting https://arxiv.org/pdf/1606.09282.pdf
    In page 6, λo is a loss balance weight, set to 1"""

def knowledge_distill_loss(full_output, prev_model, x, Temp=2.):
    beta = get_config("lwf_lambda")
    #output = model(x, full=True)
    assert T.is_tensor(full_output)
    with T.no_grad():
        prev_output = prev_model(x)
        prev_output = F.softmax(prev_output / Temp, dim=1)
    output = prev_model.process_output(full_output)
    output =  F.log_softmax(output / Temp, dim=1)
    kd_loss =  T.sum(- prev_output * output, dim=1)
    kd_loss = T.mean(kd_loss) * (Temp**2)

    return kd_loss * beta

