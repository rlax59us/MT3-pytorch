"""
Referenced original MT3 github code,
https://github.com/magenta/mt3/blob/main/mt3/network.py
"""

import torch
from typing import Any, Sequence
from data.constants import *

class Magenta_T5Config:
    vocab_size: int = VOCAB_SIZE
    """
    Token Types:
    1) Instrument(128 values)
    2) Note(128 values)
    3) On/Off(2 values)
    4) Time(205 values)
    5) Drum(128 values)
    6) End Tie Section(1 value)
    7) EOS(1 value)
    """
    dtype: Any = torch.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    head_dim: int = 64
    mlp_dim: int = 1024
    mlp_activations: Sequence[str] = ('relu',)
    dropout_rate: float = 0.1
    logits_via_embeddings: bool = False