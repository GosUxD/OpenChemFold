import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch
import importlib
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import sys

BASEDIR= '.'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = copy(importlib.import_module("cfg_2").cfg)
Net = importlib.import_module(cfg.model).Net
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_to_device(batch, device):   
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

class TestDataset(Dataset):
    def __init__(self, df):
        self.seq_map = {'A':1,'C':2,'G':3,'U':4}
        df['L'] = df.sequence.apply(len)
        self.Lmax = 206
        self.df = df
        self.padding_idx = 0   
                
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        id_min, id_max, raw_seq = self.df.loc[idx, ['id_min','id_max','sequence']]
        ids = np.arange(id_min,id_max+1)

        Lmax = len(raw_seq)
        tokens = torch.tensor([self.seq_map[s] for s in raw_seq])        
        tokens = torch.nn.functional.pad(tokens, (0, Lmax - len(tokens)), 'constant', value=self.padding_idx)
        
        ids = np.pad(ids,(0,Lmax-len(tokens)), constant_values=-1)
       
        mask_tokens = torch.ne(tokens, self.padding_idx).int()
        mask_pair_1d = torch.ne(tokens, self.padding_idx).int().unsqueeze(0)
        mask_pair_2d = mask_pair_1d * mask_pair_1d.permute(1,0)


        return {
            'tokens':tokens,
            'mask_tokens': mask_tokens,
            'mask_pair_tokens':mask_pair_2d,
            }, {'ids':ids}
        
from models.mdl_2_twintower import TwinTower
    
class Net(nn.Module):
    def __init__(self,cfg):
        super(Net, self).__init__()
        self.twintower = TwinTower(cfg)        
             
    def forward(self, batch):        
        x = self.twintower(tokens=batch['tokens'].unsqueeze(1),
                            pair_tokens=batch['tokens'],
                            mask_tokens = batch['mask_tokens'].unsqueeze(1),
                            mask_pair_tokens = batch['mask_pair_tokens']
                            )     
                 
        
        output_dict = x['chem']
        #output_dict['conf'] = compute_plddt_score(x['plddt']) / 100

        
        return output_dict 

@torch.jit.ignore
def wrapper(batch):
    tokens=batch['tokens'].unsqueeze(1)
    pair_tokens=batch['pair_tokens']
    mask_tokens = batch['mask_tokens'].unsqueeze(1)
    mask_pair_tokens = batch['mask_pair_tokens']
    
    return tokens, pair_tokens, mask_tokens, mask_pair_tokens

def collate_fn(batch):
    input = dict()
    ids = dict()
    tokens_list, mask_tokens_list, mask_pair_list, ids_list = [], [], [], []
    x, y = zip(*batch)
    batch_max_len = max([len(x['tokens']) for x in x])
    
    for data in x:
        tokens = torch.nn.functional.pad(data['tokens'], (0, batch_max_len - len(data['tokens'])), 'constant', value=0)

        mask_tokens_list.append(torch.ne(tokens, 0).int())
        tokens_list.append(tokens)
        
        mask_pair_1d = torch.ne(tokens, 0).int().unsqueeze(0)
        mask_pair_2d = mask_pair_1d * mask_pair_1d.permute(1,0)
        mask_pair_list.append(mask_pair_2d)
  
    input['tokens'] = torch.stack(tokens_list)
    input['mask_tokens'] = torch.stack(mask_tokens_list)
    input['mask_pair_tokens'] = torch.stack(mask_pair_list)
    
    for id in y:
        id_tensor = np.pad(id['ids'],(0, batch_max_len-len(id['ids'])), constant_values=-1)
        ids_list.append(torch.tensor(id_tensor))
        
    ids['ids'] = torch.stack(ids_list)
    
    return input, ids
    


id1=269545321
id2=269724007

df = pd.read_csv("datamount/test_sequences.csv")
filtered_df = df[(df['id_min'] <= id2) & (df['id_max'] >= id1)].reset_index(drop=True)

dataset = TestDataset(df)
test_dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=cfg.pin_memory,
)

weights_path = "datamount/weights/cfg_2/twintower_pretrained.pt"
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
model_state_dict = state_dict["model"]



# Remove '_orig_mod.' prefix from state_dict keys
new_state_dict = OrderedDict()
for key, value in model_state_dict.items():
    new_key = key.replace('module..', '')  # Remove the prefix
    new_state_dict[new_key] = value


model = Net(cfg).to(cfg.device)
model = torch.compile(model)
model.load_state_dict(new_state_dict)

model.eval()
torch.set_grad_enabled(False) 
ids,preds = [],[]
for x,y in (tqdm(test_dataloader, desc=f'Testing Model: ', ascii=' >=')):

    x = batch_to_device(x, cfg.device)
    y = batch_to_device(y, cfg.device)
    with autocast():
        p = model(x).clip(0,1)
        
    for idx, mask, pi in zip(y['ids'].cpu(), x['mask_tokens'].bool().cpu(), p.cpu()):
        ids.append(idx[mask])
        preds.append(pi[mask[:pi.shape[0]]])


ids = torch.concat(ids)
preds = torch.concat(preds)

df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                   'reactivity_2A3_MaP':preds[:,0].numpy()})
df.to_csv('submission_twintower.csv', index=False, float_format='%.4f') # 6.5GB