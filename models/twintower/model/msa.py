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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence, Optional

from twintower.utils.chunk_utils import chunk_layer
from twintower.model.primitives import Linear, LayerNorm

class LearnedPositionalEmbedding(nn.Embedding):
    """
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super(LearnedPositionalEmbedding, self).__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        ''' '''
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

class MSANet(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_msa = 5,
                 padding_idx = None,
                 max_len = 4096,
                 is_pos_emb = True,
                 **unused):
        super(MSANet, self).__init__()
        self.is_pos_emb = is_pos_emb

        self.embed_tokens = nn.Embedding(d_msa, d_model, padding_idx = padding_idx)
        
        # We don't need positional embedding for MSA for now
        # if self.is_pos_emb: 
        #     self.embed_positions = LearnedPositionalEmbedding(max_len, d_model, padding_idx)

    def forward(self, tokens):
        B, K, L = tokens.shape
        msa_fea = self.embed_tokens(tokens)

        if self.is_pos_emb:
            msa_fea += self.embed_positions(tokens.reshape(B * K, L)).view(msa_fea.size())

        return msa_fea

    def get_emb_weight(self):
        return self.embed_tokens.weight

class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m)

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )


    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m

