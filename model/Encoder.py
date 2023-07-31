"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Layers import *
from model.Attention import Multi_Head_Attention

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config

        self.pre_attention_layer_norm = LayerNorm(config.emb_dim)
        self.attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        
        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, inputs, encoder_mask=None, deterministic=False):
        x = self.pre_attention_layer_norm(inputs)
        x = self.attention(x, x, mask=encoder_mask, deterministic=deterministic)
        x = self.dropout1(x)
        x = x + inputs

        y = self.pre_mlp_layer_norm(x)
        y = self.mlp(y)
        y = self.dropout2(y)
        y = y + x

        return y

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        
        self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.emb_dim)
        self.embed = FixedEmbed(features=config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.layer_norm = LayerNorm(config.emb_dim)
        
    def forward(self, encoder_input_tokens, encoder_mask=None, deterministic=False):
        cfg = self.config
        assert encoder_input_tokens.ndim == 3  # [batch, length, depth]

        seq_length = encoder_input_tokens.shape[-2]
        inputs_positions = torch.arange(seq_length)[None, :].to(encoder_input_tokens.device)

        # [batch, length, depth] -> [batch, length, emb_dim]
        x = self.dense(encoder_input_tokens)
        x = x + self.embed(inputs_positions)
        x = self.dropout(x)
        x = x.type(cfg.dtype)

        for layer in self.encoder_layers:
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = layer(x, encoder_mask, deterministic)

        x = self.layer_norm(x)
        x = self.dropout(x)

        return x
