import torch
import torch.nn.functional as F


def calc_metric(cfg, val_data):
    target = val_data['target'].clip(0,1)
    predictions = val_data['predictions']
    
    loss = F.l1_loss(input=predictions, target=target,  reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss