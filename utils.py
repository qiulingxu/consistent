
MAGIC = 66

def log(*argv, **karv):
    print(*argv, **karv)

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

if "INIT_ONCE" not in globals():
    INIT_ONCE = True
    config = {}
    config["task"] = "classification"
    config["dataset"] = "cifar10"
    device = "gpu"
    set_dataset()

class PytorchModeWrap(object):
    def __init__(self, model, training):
        self.training = training
        self.model = model 

    def __enter__(self):
        self.curr_mode = self.model.training
        self.model.train(self.training)
        return self
    
    def __exit__(self):
        self.model.train(self.curr_modee)