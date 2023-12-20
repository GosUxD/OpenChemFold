import torch
import random
import os
import numpy as np
from collections import OrderedDict



def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False #initial
    torch.backends.cudnn.deterministic = True #initial
    
    os.environ['TORCHDYNAMO_REPORT_GUARD_FAILURES']='1'
    

def load_weights(weights_path):
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model_state_dict = state_dict["model"]   
    new_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        new_key = key.replace('_orig_mod.', '').replace('module.', '')  # Remove the prefix
        new_state_dict[new_key] = value
        
    return new_state_dict