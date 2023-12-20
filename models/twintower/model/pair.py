# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
from twintower.model.primitives import Linear, LayerNorm
from twintower.utils.chunk_utils import chunk_layer

 
class PairNet(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_msa = 5,
                 p_drop = 0.,
                 is_pos_emb = True,
                 padding_idx = 0
                 ):
        super(PairNet, self).__init__()

        self.pair_emb = PairEmbNet(d_model= d_model,
                                   p_drop = p_drop,
                                   d_seq  = d_msa,
                                   is_pos_emb = is_pos_emb,
                                   padding_idx = padding_idx)

    def forward(self, msa_tokens, **unused):
        seq_tokens = msa_tokens[:, 0, :]

        B, L = seq_tokens.shape
        idx = torch.cat([torch.arange(L).long().unsqueeze(0) for i in range(B)], dim=0)

        if idx.device != seq_tokens.device:
            idx = idx.to(seq_tokens.device)

        return self.pair_emb(seq_tokens, idx)

    
class RelativePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=32, p_drop=0.1):
        super(RelativePositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop)
        self.max_len = max_len
        # Set up the value bins for one-hot encoding.
        self.vbins = torch.arange(-max_len + 1, max_len).to(dtype=torch.float16, device='cuda')
        
        # Linear layer for projection after one-hot encoding.
        self.linear_proj = nn.Linear(2*max_len-1, d_model)

    def forward(self, x, idx):
        B, L, _, _ = x.shape
        #self.vbins = self.vbins.to(x.device)
        # Matrix of positions
        pos = torch.arange(L).unsqueeze(0).to(x.device)

        # Compute relative positions matrix
        rel_positions = pos.unsqueeze(-1) - pos.unsqueeze(-2)

        # Clip relative positions to [-max_len+1, max_len-1]
        rel_positions = torch.clamp(rel_positions, - self.max_len + 1, self.max_len - 1)

        # Convert the relative positions to one-hot encoding
        one_hot_rel_positions = (rel_positions.unsqueeze(-1) == self.vbins).float()

        # Project one-hot encoded relative positions
        rel_embeddings = self.linear_proj(one_hot_rel_positions)

        rel_embeddings = rel_embeddings.repeat(B, 1, 1, 1)

        x = x + rel_embeddings
        return x
        #return self.drop(x)


class PairEmbNet(nn.Module):
    def __init__(self, d_model=128, d_seq=5, p_drop=0.1,
                 is_pos_emb = True, padding_idx = 0):
        super(PairEmbNet, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb, padding_idx=padding_idx)
        self.projection = nn.Linear(d_model, d_model)

        self.is_pos_emb = is_pos_emb
        if self.is_pos_emb:
            self.pos = RelativePositionalEncoding2D(d_model, max_len=32, p_drop=p_drop)

    def forward(self, seq, idx):

        L = seq.shape[1]
        seq = self.emb(seq)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        pair = torch.cat((left, right), dim=-1)

        pair = self.projection(pair)
        pair = self.pos(pair, idx) if self.is_pos_emb else pair

        return pair


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z)

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)
        
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) * mask

        return z

    @torch.jit.ignore
    def _chunk(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )


    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)

        return z
