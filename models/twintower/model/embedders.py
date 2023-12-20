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

import torch.nn as nn
from twintower.model.msa import MSANet
from twintower.model.pair import PairNet



class MSAEmbedder(nn.Module):
    """MSAEmbedder """

    def __init__(self,
                 c_m,
                 c_z,
                 padding_idx = 0,
                 vocab_size = 5
                 ):
        super(MSAEmbedder, self).__init__()
        self.msa_emb = MSANet(d_model = c_m,
                               d_msa = vocab_size,
                               padding_idx = padding_idx,
                               is_pos_emb = False,
                               )

        self.pair_emb = PairNet(d_model = c_z,
                                 d_msa = vocab_size,
                                 padding_idx = padding_idx
                                 )

    def forward(self, tokens, pair_tokens = None, is_BKL = True, **unused):

        assert tokens.ndim == 3
        if not is_BKL:
            tokens = tokens.permute(0, 2, 1)
        msa_fea = self.msa_emb(tokens)
        pair_fea = self.pair_emb(pair_tokens.unsqueeze(1), t1ds = None, t2ds = None)

        return msa_fea, pair_fea