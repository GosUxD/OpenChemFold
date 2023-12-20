import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


#computes lddt between predictions and ground truth
def lddt(all_nucl_react_truths: Tensor, 
                   all_nucl_react_preds: Tensor, 
                   no_bins: int = 50):
    
    dist_l1 = torch.abs(all_nucl_react_truths - all_nucl_react_preds)
    dist_l1[torch.isnan(dist_l1)] = 1
    bin_width = 1 / no_bins
    bin_indices = (dist_l1 / bin_width).long()
    bin_indices = torch.clamp(bin_indices, max = no_bins - 1)
    one_hot_bins = F.one_hot(bin_indices, no_bins)
    
    return one_hot_bins

#gets the actual score in percentage from the expected value of the probability distribution    
def compute_plddt_score(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device)

    probs = F.softmax(logits, dim=-1)

    bounds = bounds.view(1, 1, 1, num_bins)

    pred_lddt_ca =  (1 - torch.sum(probs * bounds, dim=-1)) * 100

    return pred_lddt_ca 

#plddt loss for training
def plddt_loss(all_nucl_react_truths: Tensor, 
                   all_nucl_react_preds: Tensor, logits):
    
    conf_loss = nn.CrossEntropyLoss()
    
    bin_indices = lddt(all_nucl_react_truths, all_nucl_react_preds)
    bin_indices = torch.argmax(bin_indices, dim=-1)
    
    loss = conf_loss(logits.view(-1, 50), bin_indices.view(-1))
    
    return loss

#loss for training with BPP's as the pair representation auxilary task
@torch.jit.script
def bpp_loss(preds, targets, mask):
    mask = torch.sum(mask, dim=-1)[:, 0]
    loss = torch.zeros(1, device=preds.device)  # Use a tensor for accumulating loss

    batch_size = preds.size(0)

    for i in range(batch_size):
        num_elements = mask[i].item()  # Get the number of elements for this batch item

        # Select the relevant elements from preds and targets
        preds_subset = preds[i, :num_elements, :num_elements]
        targets_subset = targets[i, :num_elements, :num_elements]

        # Compute loss for this subset
        probs_subset = F.softmax(preds_subset, dim=-1)
        loss += F.cross_entropy(probs_subset, targets_subset, reduction='mean')

    # Average the loss over the batch
    loss /= batch_size
    return loss