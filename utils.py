import logging
import json

import torch.nn as nn
from typing import Union, Dict, Any
MAGIC = 66

def log(*argv, **karv):
    print(*argv, **karv)
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
    else:
        assert False

def get_config(k):
    return config[k]

VALID_OPTION = {"classification_model_process": ["mask", "mask#reset","allclass"]}

def get_config_default(k, default):
    global config
    if k not in config:
        config[k] =default
        return default
    else:
        return config[k]

def set_config(k, v):
    global config
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
    
    #set_dataset()

class PytorchModeWrap(object):
    def __init__(self, model :Union[nn.Module,Dict[str,nn.Module]], training):
        self.training = training
        self.model = model 

    def __enter__(self):
        
        if isinstance(self.model, dict):
            self.curr_mode = {}
            for k,m in self.model.items():
                self.curr_mode[k] = m.training
                m.train(self.training)
        else:
            self.curr_mode = self.model.training
            self.model.train(self.training)
        return self
    
    def __exit__(self,exception_type, exception_value, traceback):
        if isinstance(self.model, dict):
            for k,m in self.model.items():
                m.train(self.curr_mode[k])
        else:
            self.model.train(self.curr_mode)

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