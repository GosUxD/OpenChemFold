import pandas as pd
import os, gc
import numpy as np
from tqdm import tqdm
import math
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast
import torch
import importlib
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional
import typing
from torch import Tensor
from types import SimpleNamespace
from collections import OrderedDict

import os
import sys

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

def string_to_list(string):
    try:
        # Remove brackets and split the string
        elements = string.strip('[]').split(',')
        # Convert elements to float, handling 'nan' appropriately
        return [float(el.strip()) if el.strip().lower() != 'nan' else float('nan') for el in elements]
    except ValueError as e:
        print("Error:", e)
        print("String causing error:", string)
        return None  # or a default value

def generate_oof_targets(preds: Tensor, reacts: Tensor, confidence: Tensor):

    linear_transform_conf = 1.2 * (confidence - 0.5) + 0.1
    scaled_confidence = torch.clamp(linear_transform_conf, min=0.1, max=0.7)
    
    weight_pred = scaled_confidence
    weight_truth = 1 - scaled_confidence
    
    combined_reactivity = 0.65 * preds + 0.35 * reacts
    
    return combined_reactivity


BASEDIR= '.'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = copy(importlib.import_module("cfg_2").cfg)
Net = importlib.import_module(cfg.model).Net
compute_plddt_score = importlib.import_module("loss_utils").compute_plddt_score
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_to_device(batch, device):   
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

class TestDataset(Dataset):
    def __init__(self, df, aug=None, training = True, fold=0, seed=2023, nfolds=4):
        self.training = training
        self.seq_map = {'A':1,'C':2,'G':3,'U':4}
        self.Lmax = 206
        self.padding_idx = 0
        self.mask_index = 5
                
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        
        #We can vary the signal to noise threshold here
        m = (df_2A3['signal_to_noise'].values > 0.35) & (df_DMS['signal_to_noise'].values > 0.35)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        self.seq_id = df_2A3['sequence_id'].values
        
        self.react_2A3 = df_2A3['target']                         
        self.react_DMS = df_DMS['target']                       
        self.react_err_2A3 = df_2A3['target_error']                        
        self.react_err_DMS = df_DMS['target_error']            
        
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        raw_seq = self.seq[idx]
        seq_id = self.seq_id[idx]
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        
        
        tokens = torch.tensor([self.seq_map[s] for s in raw_seq])
        target_labels = tokens.clone()
        pair_tokens = tokens.clone()
        #if self.training:
        mask_probabilities = torch.rand(tokens.size(), dtype=torch.float32) < 0.00
        tokens[mask_probabilities] = self.mask_index
        target_labels[~mask_probabilities] = -100

        
        tokens = torch.nn.functional.pad(tokens, (0, 206 - len(tokens)), 'constant', value=self.padding_idx)
        pair_tokens = torch.nn.functional.pad(pair_tokens, (0, 206 - len(pair_tokens)), 'constant', value=self.padding_idx)
        target_labels = torch.nn.functional.pad(target_labels, (0, 206-len(target_labels)), 'constant', value=-100)
        
        
        
        mask_tokens = torch.ne(tokens, self.padding_idx).int()
        mask_pair_1d = torch.ne(tokens, self.padding_idx).int().unsqueeze(0)
        mask_pair_2d = mask_pair_1d * mask_pair_1d.permute(1,0)

 
        return {
            'tokens':tokens,
            'pair_tokens':pair_tokens,
            'target_labels':target_labels,
            'mask_tokens': mask_tokens,
            'mask_pair_tokens':mask_pair_2d,
            'target':react, 
            'target_err':react_err,
            }, {"sequence": self.seq[idx],
                'sequence_id':seq_id}

from models.mdl_2_twintower import TwinTower

    
class Net(nn.Module):
    def __init__(self,cfg):
        super(Net, self).__init__()
        self.twintower = TwinTower(cfg)      

    def forward(self, batch):        
        x = self.twintower(tokens=batch['tokens'].unsqueeze(1),
                            pair_tokens=batch['pair_tokens'],
                            mask_tokens = batch['mask_tokens'].unsqueeze(1),
                            mask_pair_tokens = batch['mask_pair_tokens']
                            )     
                 
        output_dict = {}
        
        output_dict['pred'] = x['chem']
        output_dict['conf'] = compute_plddt_score(x['plddt']) / 100

        
        return output_dict 


df = pd.read_parquet("datamount/train_data(not_passed_SN_filter).parquet")

dataset = TestDataset(df)
print(len(dataset))
test_dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=1,
        pin_memory=cfg.pin_memory,
)

weights_path = "datamount/weights/cfg_2/twintower_pretrained.pt"
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
model_state_dict = state_dict["model"]

# Remove '_orig_mod.' prefix from state_dict keys
new_state_dict = OrderedDict()
for key, value in model_state_dict.items():
    new_key = key.replace('module.', '')  # Remove the prefix #.replace('_orig_mod.', '').
    new_state_dict[new_key] = value

model = Net(cfg).to(cfg.device)
model = torch.compile(model)
model.load_state_dict(new_state_dict)
torch.set_printoptions(precision=5, sci_mode=False)

model.eval()
torch.set_grad_enabled(False) 
sequences, sequence_ids, targets, errors, experiment_types = [],[],[],[],[]
for i, (x, y) in enumerate(tqdm(test_dataloader, desc=f'Generating Synthetic Data: ', ascii=' >=')):

    x = batch_to_device(x, cfg.device)
    sequence = y['sequence'][0]
    sequence_id = y['sequence_id'][0]
    with autocast():
        p = model(x)
        
    pred_2A3 = p['pred'][:, :, 0]
    conf_2A3 = p['conf'][:, :, 0]
    react_2A3 = x['target'][:, :, 0]
    error_2A3 = x['target_err'][:, :, 0]

    combined_2A3 = generate_oof_targets(preds=pred_2A3, reacts=react_2A3, confidence=conf_2A3)


    pred_DMS = p['pred'][:, :, 1]
    conf_DMS = p['conf'][:, :, 1]   
    react_DMS = x['target'][:, :, 1]
    error_DMS = x['target_err'][:, :, 1]
    
    combined_DMS = generate_oof_targets(preds=pred_DMS, reacts=react_DMS, confidence=conf_DMS)

    
    sequence_ids.append(sequence_id)
    sequences.append(sequence)
    targets.append(combined_2A3.squeeze(0).tolist())
    errors.append(error_2A3.squeeze(0).tolist())
    experiment_types.append("2A3_MaP")
    
    sequence_ids.append(sequence_id)
    sequences.append(sequence)
    targets.append(combined_DMS.squeeze(0).tolist())
    errors.append(error_DMS.squeeze(0).tolist())
    experiment_types.append("DMS_MaP")

    

synthetic_df = pd.DataFrame({'sequence': sequences,
                            'sequence_id':sequence_ids,
                            'experiment_type':experiment_types,
                            'target': targets,
                            'target_error': errors})

synthetic_df.to_parquet("datamount/synthetic_data_.parquet", compression='gzip')



