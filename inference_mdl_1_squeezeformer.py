import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional
from torch import Tensor
from types import SimpleNamespace
from collections import OrderedDict
import os
import sys

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

class CustomDataset(Dataset):
    def __init__(self, df, mask_only=False, **kwargs):
        self.bpp_paths_df = pd.read_csv("datamount/bpp_index.csv")
        self.bpp_paths = dict(zip(self.bpp_paths_df['sequence_id'], self.bpp_paths_df['file_path']))
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        self.df = df
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        id_min, id_max, seq, seq_id = self.df.loc[idx, ['id_min','id_max','sequence','sequence_id']]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        if self.mask_only: return {'mask':mask},{}
        ids = np.arange(id_min,id_max+1)
        
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        seq = np.pad(seq,(0,self.Lmax-L))
        ids = np.pad(ids,(0,self.Lmax-L), constant_values=-1)
        
        npy_path = self.bpp_paths[seq_id]
        base_pair_probabilities = np.load(npy_path)['arr_0'].astype(np.float32)
        base_pair_probabilities = np.pad(
            base_pair_probabilities,(
                (0,self.Lmax-base_pair_probabilities.shape[0]),
                (0,self.Lmax-base_pair_probabilities.shape[1])
            )
        )
        assert base_pair_probabilities.shape[0] == self.Lmax
        assert base_pair_probabilities.shape[1] == self.Lmax

        
        return {'input':torch.from_numpy(seq), 'input_mask':mask, 'bpp':base_pair_probabilities}, \
               {'ids':ids} 

def batch_to_device(batch, device):   
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
        
class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        
        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        
        return x
  
class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
    
class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
        bpp_dim = 206
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)
        
        
        self.bpp_transform2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),  # Example parameters
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding='same'),
        )
 
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
        bpp: Tensor = None
    ) -> Tensor:
        batch_size = value.size(0)

        bpp = bpp.unsqueeze(1)
        bpp_bias = self.bpp_transform2(bpp)
        bpp_bias = bpp_bias.expand(-1, self.num_heads, -1, -1)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        
        
        score = (content_score + pos_score) / self.sqrt_dim + bpp_bias

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)
    
    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score
    
class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None, bpp: Tensor = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask, bpp=bpp)        

        return self.dropout(outputs)
        
class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class DepthwiseConv2d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 2-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    # mask_pad = mask.bool().unsqueeze(1)
    def forward(self, x, mask_pad):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        Reference for masking : https://github.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch/wenet/transformer/convolution.py#L26
        """
        # mask batch padding
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        # torch.Size([4, 128, 384])
        x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
        mask_bn = mask_pad.view(-1)
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        '''
        x = self.bn(x)
        '''
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x
   
class FoldformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(FoldformerBlock, self).__init__()
        
        # self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        # self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        # self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        # self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)
        
        self.mhsa = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,)
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)


    def forward(self, x, mask, bpp: Tensor):
        mask_pad = ( mask).long().bool().unsqueeze(1)
        mask_pad = ~( mask_pad.permute(0, 2,1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape
        
        residual = x
        #x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa(x, mask_pad, bpp)
        # Skip pad #1
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        
        x = self.ln_mhsa(x)

        residual = x
        #x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)
        
        
        # Unskip pad #1
        x_skip[mask_flat] = x[0]
        x = x_skip.view(bs, slen, nfeats)
        residual = x
        # torch.Size([16, 384, 128])
        #x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, mask_pad = mask.bool().unsqueeze(1))
        # Skip pad #2
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        
        x = self.ln_conv(x)
        
        
        residual = x
        #x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)
        
        # Unskip pad #2
        x_skip[mask_flat] = x[0]
        x = x_skip.view(bs, slen, nfeats)  
        
        
        return x
    
class FoldformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(FoldformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                FoldformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

    def forward(self, x: Tensor, mask: Tensor, bpp: Tensor):
        for idx, block in enumerate(self.blocks):
            x = block(x, mask, bpp)

        return x

def loss(pred,batch):
    p = pred[batch['input_mask'][:,:pred.shape[1]]]
    y = batch['target'][batch['input_mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss
        
class Net(nn.Module):
    def __init__(self,cfg):
        super(Net, self).__init__()
        
        self.max_sequence = cfg.max_sequence
        self.embed = nn.Embedding(4, cfg.d_model)
        
        self.encoder = FoldformerEncoder(
                      input_dim=cfg.encoder_config.input_dim,
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      input_dropout_p= cfg.encoder_config.input_dropout_p,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
        
        self.out_proj = nn.Linear(cfg.d_model, 2)
        self.loss_fn = loss
        print(f"n_params: {count_parameters(self):_}")
        
    
    def forward(self, batch):
        x = batch['input']
        bpp = batch['bpp']

        mask = batch['input_mask'].long()

        x = self.embed(x)
        x = self.encoder(x, mask, bpp)
        x = self.out_proj(x)
        
        return x 
    

cfg = SimpleNamespace(**{})

cfg.max_sequence = 206
cfg.batch_size = 256
cfg.num_workers = 4
cfg.pin_memory = False
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

cfg.d_model = 192

encoder_config = SimpleNamespace(**{})
encoder_config.input_dim=192
encoder_config.encoder_dim=192
encoder_config.num_layers=14
encoder_config.num_attention_heads= 6
encoder_config.feed_forward_expansion_factor=4
encoder_config.conv_expansion_factor= 2
encoder_config.input_dropout_p= 0.1
encoder_config.feed_forward_dropout_p= 0.1
encoder_config.attention_dropout_p= 0.1
encoder_config.conv_dropout_p= 0.1
encoder_config.conv_kernel_size= 51

cfg.encoder_config = encoder_config
    
id1=269545321
id2=269724007

df = pd.read_csv("datamount/test_sequences.csv")
#Filtered dataframe only for MutateAndMap visualization
filtered_df = df[(df['id_min'] <= id2) & (df['id_max'] >= id1)].reset_index(drop=True)

dataset = CustomDataset(df)
test_dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=cfg.pin_memory,
)

weights_path = "datamount/weights/cfg_1/squeezeformer_pretrained.pt"
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
model_state_dict = state_dict["model"]

# Remove '_orig_mod.' prefix from state_dict keys
new_state_dict = OrderedDict()
for key, value in model_state_dict.items():
    new_key = key.replace('module.', '')  # Remove the prefix
    new_state_dict[new_key] = value

model = Net(cfg).to(cfg.device)
model = torch.compile(model)
model.load_state_dict(new_state_dict)

model.eval()
torch.set_grad_enabled(False) 
ids,preds = [],[]

for x,y in (tqdm(test_dataloader, desc=f'Model Test: ', ascii=' >=')):
      
    x = batch_to_device(x, cfg.device)
    y = batch_to_device(y, cfg.device)
    with autocast():
        p = model(x).clip(0,1)
        
    for idx, mask, pi in zip(y['ids'].cpu(), x['input_mask'].cpu(), p.cpu()):
        ids.append(idx[mask])
        preds.append(pi[mask[:pi.shape[0]]])

ids = torch.concat(ids)
preds = torch.concat(preds)

df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                   'reactivity_2A3_MaP':preds[:,0].numpy()})
df.to_csv('submission_squeezeformer.csv', index=False, float_format='%.4f') # 6.5GB
