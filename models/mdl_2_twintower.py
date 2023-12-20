import torch 
import torch.nn as nn

from torch.nn import functional as F
from twintower.utils.tensor_utils import add
from twintower.model.chemformer import ChemformerStack
from twintower.model.embedders import MSAEmbedder
from twintower.model.heads import pLDDTHead, ChemMSAHead, SSHead
from metrics.loss_utils import plddt_loss, compute_plddt_score, bpp_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TwinTower(nn.Module):
    def __init__(self, cfg):
        
        super(TwinTower, self).__init__()
        
        self.cfg = cfg
        
        self.msa_embedder = MSAEmbedder(
            c_m=cfg.msa_embedder.c_m,
            c_z=cfg.msa_embedder.c_z,
            padding_idx=cfg.padding_index,
            vocab_size=cfg.vocab_size
        )
        
        self.chemformer = ChemformerStack(
            c_m = cfg.chemformer_stack.c_m,
            c_z = cfg.chemformer_stack.c_z,
            c_hidden_msa_att = cfg.chemformer_stack.c_hidden_msa_att,
            c_hidden_opm = cfg.chemformer_stack.c_hidden_opm,
            c_hidden_mul = cfg.chemformer_stack.c_hidden_mul, 
            c_hidden_pair_att = cfg.chemformer_stack.c_hidden_pair_att,
            c_s=cfg.chemformer_stack.c_s,
            no_heads_msa = cfg.chemformer_stack.no_heads_msa,
            no_heads_pair = cfg.chemformer_stack.no_heads_pair,
            no_blocks = cfg.chemformer_stack.no_blocks,
            transition_n = cfg.chemformer_stack.transition_n,
            blocks_per_ckpt = cfg.chemformer_stack.blocks_per_ckpt
        )         
        
        # self.ss_head = SSHead(
        #     c_in= cfg.bpp_head.c_in,
        # )
         
        self.chem_head = ChemMSAHead(
            no_bins=2,
            c_in=cfg.ss_head.c_in,
            c_hidden=cfg.ss_head.c_hidden
        )
        
        self.plddt_head = pLDDTHead(
            c_in=cfg.plddt_head.c_in,
            no_bins=cfg.plddt_head.no_bins
        )
        print(f"n_params: {count_parameters(self):_}")
    
    def forward_one_cycle(self, tokens, pair_tokens, mask_tokens, mask_pair_tokens, recycling_inputs=None, ):
        """
            tokens [batch, seq_len, channels]
            pair_tokens [batch, seq_len, channels]
        """ 

        msa_features, pair_features = self.msa_embedder.forward(tokens=tokens,
                                                               pair_tokens=pair_tokens,
                                                               is_BKL=True)
        
        msa_features, pair_features, single_feature = self.chemformer(
            m=msa_features,
            z=pair_features,
            msa_mask=mask_tokens,
            pair_mask=mask_pair_tokens,
            chunk_size=None
        )
        output = {}               
        output['chem'] = self.chem_head(single_feature)
        output['plddt'] = self.plddt_head(single_feature)
        #output['bpp'] = self.ss_head(pair_features.float())
        
        return output
        
    
    def forward(self, tokens, pair_tokens, mask_tokens, mask_pair_tokens, **kwargs):
        recycling_inputs = None
        output = None
        
        output = self.forward_one_cycle(tokens=tokens, 
                                        pair_tokens=pair_tokens, 
                                        recycling_inputs=recycling_inputs, 
                                        mask_tokens=mask_tokens,
                                        mask_pair_tokens=mask_pair_tokens)

        return output
    
@torch.jit.script
def loss(pred, target, mask):
    p = pred[mask[:,:pred.shape[1]]]
    y = target[mask].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss

    
class Net(nn.Module):
    def __init__(self,cfg):
        super(Net, self).__init__()
        self.twintower = TwinTower(cfg)
        self.mae_loss_fn = loss
        #self.bpp_loss = bpp_loss
        self.confid_loss = plddt_loss
        
              
    def forward(self, batch):      
        x = self.twintower(*wrapper(batch))     

        loss_mae = self.mae_loss_fn(x['chem'], batch['target'], batch['mask_tokens'].bool())
        loss_confidence = self.confid_loss(batch['target'], x['chem'], x['plddt'])
        #Submission was without BPP loss
        #loss_bpp = self.bpp_loss(x['bpp'], batch['bpp'], batch['mask_pair'])
        
        loss = loss_mae + (0.01 * loss_confidence) #+ (0.4 * loss_bpp)
        confidence = compute_plddt_score(x['plddt'])
         
        output_dict = {
            "loss": loss,
            #"loss_bpp": loss_bpp,
            "loss_mae": loss_mae,
            "loss_conf": loss_confidence,
            "target": batch['target'],
            "predictions": x['chem'],
            "confidence": confidence
            }
        
        return output_dict 

@torch.jit.ignore
def wrapper(batch):
    tokens=batch['tokens'].unsqueeze(1)
    pair_tokens=batch['pair_tokens']
    mask_tokens = batch['mask_tokens'].unsqueeze(1)
    mask_pair_tokens = batch['mask_pair']
    
    return tokens, pair_tokens, mask_tokens, mask_pair_tokens