"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
import torch.nn as nn

from model.Layers import *
from model.Mask import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, head_dim, dtype=torch.float32, dropout_rate=0.0, kernel_init=None, float32_logits=False):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.projection = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_
        self.float32_logits = float32_logits
        self.output = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim)

    def dot_product_attention(self, query, key, value, bias=None, deterministic=False):
        assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], ('q, k, v batch dims must match.')
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], ('q, k, v num_heads must match.')
        assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # Casting logits and softmax computation for float32 for model stability.
        if self.float32_logits:
            query = query.float()
            key = key.float()

        # `attn_weights`: [batch, num_heads, q_length, kv_length]
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key)

        # Apply attention bias: masking, dropout, proximity bias, etc.
        if bias is not None:
            attn_weights = attn_weights + bias.to(attn_weights.dtype).to(device)

        # Normalize the attention weights across `kv_length` dimension.
        attn_weights = F.softmax(attn_weights, dim=-1).to(self.dtype)

        attn_weights = self.dropout(attn_weights) #edited from original code

        # Take the linear combination of `value`.
        return torch.einsum('bhqk,bkhd->bqhd', attn_weights, value)

    def forward(self, inputs_q, inputs_kv, mask=None, bias=None, decode=False, deterministic=False):
        #In Original MT3, they initialize the parameter with query_init, using customized Dense Layer
        query = self.projection(inputs_q).view(inputs_q.size(0), inputs_q.size(1), self.num_heads, self.head_dim)
        key = self.projection(inputs_kv).view(inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim)
        value = self.projection(inputs_kv).view(inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim)

        if decode:
            is_initialized = hasattr(self, 'cached_key')
            swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
            cached_key = getattr(self, 'cached_key', torch.zeros(*swap_dims(key.size()), dtype=key.dtype, device=key.device))
            cached_value = getattr(self, 'cached_value', torch.zeros(*swap_dims(value.size()), dtype=value.dtype, device=value.device))
            cache_index = getattr(self, 'cache_index', torch.tensor(0, dtype=torch.int32))
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.size()
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.size():
                    raise ValueError('Autoregressive cache shape error, '
                                    'expected query shape %s instead got %s.' %
                                    (expected_shape, query.size()))

                cur_index = cache_index.item()
                one_hot_indices = F.one_hot(cur_index, length).type(key.dtype)
                one_token_key = key.permute(0, 2, 1, 3)
                one_token_value = value.permute(0, 2, 1, 3)
                key = cached_key + one_token_key * one_hot_indices.unsqueeze(-1)
                value = cached_value + one_token_value * one_hot_indices.unsqueeze(-1)
                setattr(self, 'cached_key', key)
                setattr(self, 'cached_value', value)
                setattr(self, 'cache_index', cache_index + 1)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)

                mask = combine_masks(
                    mask,
                    (torch.arange(length, device=mask.device) <= cur_index).unsqueeze(0).unsqueeze(0))

                if bias is not None:
                    bias = bias.squeeze(0)[cur_index].unsqueeze(0).unsqueeze(-2)

        if mask is not None:
            attention_bias = torch.where(mask > 0,
                                         torch.zeros_like(mask).to(self.dtype),
                                         -1e10 * torch.ones_like(mask).to(self.dtype))
        else:
            attention_bias = None

        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)
        
        x = self.dot_product_attention(
                query,
                key,
                value,
                bias=attention_bias,
                deterministic=deterministic)
        
        out = self.output(x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3)))
        
        return out
