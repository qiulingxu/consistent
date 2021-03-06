import logging
import json

import torch.nn as nn
from typing import Union, Dict, Any
import numpy as np

MAGIC = 66

def log(*argv, **karv):
    #print(*argv, **karv)
    logger.info(" ".join([str(_) for _ in argv]))

def assert_keys_in_dict(keys, dct):
    for key in keys:
        assert key in dct, "key {} not in dictionary".format(key)
    return True

def set_dataset():
    global config
    assert "task" in config
    assert "dataset" in config
    if config["dataset"] == "cifar10":
        config["IMG_SIZE"] = (3, 32, 32)
        config["CLASS_NUM"] = 10
    elif config["dataset"] == "cifar100":
        config["IMG_SIZE"] = (3, 32, 32)
        config["CLASS_NUM"] = 100
    elif config["dataset"] == "imagenet32":
        config["IMG_SIZE"] = (3, 32, 32)
        config["CLASS_NUM"] = 1000
    elif config["dataset"] == "stl10":
        config["IMG_SIZE"] = (3, 96, 96)
        config["CLASS_NUM"] = 10
    else:
        assert False

def get_config(k):
    return config[k]

VALID_OPTION = {"classification_model_process": ["mask", "mask#reset","allclass"]}

def get_config_default(k, default):
    global config
    if k not in config:
        set_config(k, default)
        return default
    else:
        return config[k]

def set_config(k, v):
    global config
    if k in VALID_OPTION:
        assert v in VALID_OPTION[k]
    config[k] = v

def save_config(path):
    with open(path + "_config.json", "w") as f:
        f.write(json.dumps(config))
        
if "INIT_ONCE" not in globals():
    INIT_ONCE = True
    config = {} # type: Dict[str, Any]
    #config["task"] = "classification"
    #config["dataset"] = "cifar10"
    device = "cuda"
    debug = True
    logging.basicConfig(filename='./logs/app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger=logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    #set_dataset()

def get_fixed_random_index(num, seed):
    np.random.seed(seed)
    shuffle = np.random.permutation(num)
    return shuffle

def get_fixed_random_generator(num, seed, prob=None):
    np.random.seed(seed)
    l = list(range(num))
    while True:
        yield np.random.choice(l,p=prob)
        #np.random.randint(num)

def freeze(model: nn.Module):
    var_list =[] 
    for name, param in model.named_parameters():
        if param.requires_grad:
            var_list.append(name)
        param.requires_grad = False
    return var_list
class PytorchModeWrap(object):
    def __init__(self, model :Union[nn.Module,Dict[str,nn.Module]], training):
        self.training = training
        assert training in [True, False]
        self.model = model 

    def __enter__(self):
        
        if isinstance(self.model, dict):
            self.curr_mode = {}
            for k,m in self.model.items():
                assert m.training in [True, False]
                self.curr_mode[k] = m.training
                m.train(self.training)
        else:
            assert self.model.training in [True, False]
            self.curr_mode = self.model.training
            self.model.train(self.training)
        return self
    
    def __exit__(self,exception_type, exception_value, traceback):
        if isinstance(self.model, dict):
            for k,m in self.model.items():
                m.train(self.curr_mode[k])
        else:
            self.model.train(self.curr_mode)

class PytorchFixWrap(object):
    def __init__(self, model :nn.Module, var_list, requires_grad):
        self.requires_grad = requires_grad
        self.var_list = set(var_list)
        self.model = model 
        self.prev_stat = {}

    def __enter__(self):
        
        param = self.model.named_parameters()
        for k,v in param:
            if k in self.var_list:
                self.prev_stat[k] = v.requires_grad
                v.requires_grad = self.requires_grad
        return self
    
    def __exit__(self,exception_type, exception_value, traceback):
        param = self.model.named_parameters()
        for k,v in param:
            if k in self.var_list:
                v.requires_grad = self.prev_stat[k]

def get_key_default(dct, key, default, range = None, type = None):
    if key in dct:
        val = dct[key]
    else:
        val = default
    if range is not None:
        assert val in range, "Value {} is not in Range {}".format(val, range)
    if type is not None:
        assert isinstance(val, type), "Value {} is not Typr {}".format(val, type)
    return val

def dict_index_range(dct, start, end):
    lst = []
    for i in range(start, end):
        lst.append(dct[i])
    return lst

def repeat_dataloader(loader):
    while True:
        for idx, data in enumerate(loader):
            yield data