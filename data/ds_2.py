import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import sys

def batch_to_device(batch, device):   
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

       
class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode = 'train', df_synthetic=None, fold=0, seed=2023, nfolds=4):
        self.mode = mode
        self.bpp_paths_df = pd.read_csv("datamount/bpp_index.csv")
        self.bpp_paths = dict(zip(self.bpp_paths_df['sequence_id'], self.bpp_paths_df['file_path']))
        self.seq_map = {'A':1,'C':2,'G':3,'U':4}
        self.Lmax = 206
        self.padding_idx = 0
        self.mask_index = 5
                
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        if mode == "train":
            df_2A3 = df_2A3[df_2A3['fold'] != fold].reset_index(drop=True)
            df_DMS = df_DMS[df_DMS['fold'] != fold].reset_index(drop=True)
        elif mode == "val":
            df_2A3 = df_2A3[df_2A3['fold'] == fold].reset_index(drop=True)
            df_DMS = df_DMS[df_DMS['fold'] == fold].reset_index(drop=True)
        else:
            df_2A3 = df_2A3.reset_index(drop=True)
            df_DMS = df_DMS.reset_index(drop=True)
        
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        if df_synthetic is not None:
            df_2A3_synth = df_synthetic[df_synthetic['experiment_type'] == "2A3_MaP"]
            df_DMS_synth = df_synthetic[df_synthetic['experiment_type'] == "DMS_MaP"]
            
            df_2A3 = pd.concat([df_2A3, df_2A3_synth], ignore_index=True, axis=0)
            df_DMS = pd.concat([df_DMS, df_DMS_synth], ignore_index=True, axis=0)
        
        self.seq = df_2A3['sequence'].values
        self.seq_id = df_2A3['sequence_id'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3['target']                         
        self.react_DMS = df_DMS['target']                       
        self.react_err_2A3 = df_2A3['target_error']                        
        self.react_err_DMS = df_DMS['target_error']            
        
        
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
        
        #For potential MLM training, currently mask probabilities is 0.00
        mask_probabilities = torch.rand(tokens.size(), dtype=torch.float32) < 0.00
        tokens[mask_probabilities] = self.mask_index
        target_labels[~mask_probabilities] = -100

        
        tokens = torch.nn.functional.pad(tokens, (0, 206 - len(tokens)), 'constant', value=self.padding_idx)
        pair_tokens = torch.nn.functional.pad(pair_tokens, (0, 206 - len(pair_tokens)), 'constant', value=self.padding_idx)
        target_labels = torch.nn.functional.pad(target_labels, (0, 206-len(target_labels)), 'constant', value=-100)
        
        
        
        mask_tokens = torch.ne(tokens, self.padding_idx).int()
        mask_pair_1d = torch.ne(tokens, self.padding_idx).int().unsqueeze(0)
        mask_pair_2d = mask_pair_1d * mask_pair_1d.permute(1,0)
        
        npy_path = self.bpp_paths[seq_id]
        base_pair_probabilities = np.load(npy_path)['arr_0'].astype(np.float16)
        base_pair_probabilities = torch.from_numpy(np.pad(
            base_pair_probabilities,(
                (0,self.Lmax-base_pair_probabilities.shape[0]),
                (0,self.Lmax-base_pair_probabilities.shape[1])
            )
        ))

        assert base_pair_probabilities.shape[0] == self.Lmax
        assert base_pair_probabilities.shape[1] == self.Lmax


        return {
            'tokens':tokens,
            'pair_tokens':pair_tokens,
            'target_labels':target_labels,
            'mask_tokens': mask_tokens,
            'mask_pair':mask_pair_2d,
            'target':react, 
            'target_err':react_err,
            'bpp': base_pair_probabilities,
            }
