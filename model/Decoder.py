"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Layers import *
from model.Attention import Multi_Head_Attention

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.config = config

        self.pre_self_attention_layer_norm = LayerNorm(config.emb_dim)
        self.self_attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.pre_cross_attention_layer_norm = LayerNorm(config.emb_dim)
        self.encoder_decoder_attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        self.dropout3 = nn.Dropout(config.dropout_rate)

    def forward(self, inputs, encoded, decoder_mask=None, encoder_decoder_mask=None, decode=False):
        x = self.pre_self_attention_layer_norm(inputs)
        x= self.self_attention(x, x, mask=decoder_mask)
        x = self.dropout1(x)
        x = x + inputs

        y = self.pre_cross_attention_layer_norm(x)
        y = self.encoder_decoder_attention(y, encoded, mask=encoder_decoder_mask)
        y = self.dropout2(y)
        y = y + x

        z = self.pre_mlp_layer_norm(y)
        z = self.mlp(z)
        z = self.dropout3(z)
        z = z + y

        return z

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        
        self.token_embed = Embed(config.vocab_size, config.emb_dim, one_hot=True)
        self.fixed_embed = FixedEmbed(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.layer_norm = LayerNorm(config.emb_dim)
        self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        
    def forward(self, encoded, decoder_input_tokens, decoder_positions=None, decoder_mask=None, encoder_decoder_mask=None, deterministic=False, decode=False, max_decode_length=None):
        cfg = self.config
        assert decoder_input_tokens.ndim == 2  # [batch, len]

        seq_length = decoder_input_tokens.shape[-1]
        decoder_positions = torch.arange(seq_length)[None, :].to(decoder_input_tokens.device)

        # [batch, length] -> [batch, length, emb_dim]
        y = self.token_embed(decoder_input_tokens)
        y = y + self.fixed_embed(decoder_positions, decode=decode)
        y = self.dropout(y)
        y = y.type(cfg.dtype)

        for layer in self.decoder_layers:
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            y = layer(y, encoded, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask, decode=decode)

        y = self.layer_norm(y)
        y = self.dropout(y)

        # [batch, length, emb_dim] -> [batch, length, vocab_size]
        logits = self.dense(y)

        return logits
