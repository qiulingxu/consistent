from ..taskdata import Seq_IDomain_CD, Con_IDomain_CD, Seq_IData_CD, Con_IData_CD
from ..utils import get_config_default, set_config

def cifar10_incremental_config():
    ret = {}
    dataset = get_config_default("dataset", "cifar10")
    converge_decay = get_config_default("convergence_decay_rate", 0.9)
    converge_thresh = get_config_default("convergence_improvement_threshold", 1e-3)
    class_inc_mode = get_config_default("classification_model_process", "mask")
    classification_task = get_config_default("classification_task","domain_inc")
    develop_assumption = get_config_default("develop_assumption","sequential")
    domain_inc_parameter = {"segments":5,"batch_size":128}
    assert develop_assumption in ["sequential", "concurrent"]
    assert classification_task in ["domain_inc", "data_inc"]
    if classification_task == "domain_inc":
        domain_inc_parameter = get_config_default("ic_parameter", domain_inc_parameter)
        if develop_assumption == "sequential":
            taskdata = Seq_IDomain_CD
        else:
            taskdata = Con_IDomain_CD
    elif classification_task == "data_inc":
        domain_inc_parameter = get_config_default("ic_parameter", domain_inc_parameter)
        if develop_assumption == "sequential":
            taskdata = Seq_IData_CD
        else:
            taskdata = Con_IData_CD
    full_name = "DS{}_CIM{}_CT{}_DA{}_CvgD{:.2e}_CvgT{:2e}_DomS{}".format(dataset,\
                                                class_inc_mode,\
                                                classification_task,\
                                                develop_assumption,\
                                                converge_decay,\
                                                converge_thresh,\
                                                domain_inc_parameter["segments"]
                                                )
    set_config("full_name", full_name)
    ret["taskdata"] = taskdata
    return ret