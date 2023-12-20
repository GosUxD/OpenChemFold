import torch
import torch.nn as nn
from twintower.utils import default
from twintower.model.primitives import Linear, LayerNorm

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2

        return x

class FeedForwardLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 p_drop = 0.1,
                 d_model_out = None,
                 is_post_act_ln = False,
                 **unused,
                 ):

        super(FeedForwardLayer, self).__init__()
        d_model_out = default(d_model_out, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.post_act_ln = LayerNorm(d_ff) if is_post_act_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model_out)
        self.activation = nn.ReLU()

    def forward(self, src):
        src = self.linear2(self.dropout(self.post_act_ln(self.activation(self.linear1(src)))))
        return src
    
class SSHead(nn.Module):
    def __init__(self,
                 c_in,
                 no_bins=1,
                 **kwargs):
        super(SSHead, self).__init__()
        self.norm = LayerNorm(c_in)
        self.proj = nn.Linear(c_in, c_in)
        self.ffn = FeedForwardLayer(d_model=c_in, d_ff = c_in*4, d_model_out=no_bins, **kwargs)

    def forward(self, x):

        x = self.norm(x)
        x = self.proj(x)
        x = 0.5 * (x + x.permute(0, 2, 1, 3))
        logits = self.ffn(x).squeeze(-1)

        return logits
    
    
class pLDDTHead(nn.Module):
    def __init__(self, c_in, no_bins = 50):
        super(pLDDTHead, self).__init__()

        self.no_bins = no_bins
        
        self.net_lddt = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, no_bins * 2),
        )
        self.sfmx = nn.Softmax(dim=2)
 
    def forward(self, sfea_tns):

        logits = self.net_lddt(sfea_tns)
        bs, seq = logits.shape[0], logits.shape[1]
        
        logits = logits.view(bs, seq, 2, self.no_bins)

        return  logits

class ChemMSAHead(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super(ChemMSAHead, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden)
        self.linear_2 = Linear(self.c_hidden, self.c_hidden)
        self.linear_3 = Linear(self.c_hidden, self.no_bins)

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s