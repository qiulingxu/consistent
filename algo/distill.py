import torch.nn as nn
import torch as T

l2_loss = nn.MSELoss()
def knowledge_distill_loss(model, prev_model, x):
    output = model(x, full=True)
    prev_output = prev_model(x)
    output = prev_model.process_output(output)
    return l2_loss(output - prev_output)