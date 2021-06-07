

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
    config = object()
    config["task"] = "classification"
    config["dataset"] = "cifar10"
    set_dataset()