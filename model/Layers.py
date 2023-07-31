"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def sinusoidal(shape, min_scale: float = 1.0,
               max_scale: float = 10000.0,
               dtype = torch.float32):
    """Sinusoidal init."""
    if dtype != torch.float32:
        raise ValueError('The sinusoidal initializer only supports float32.')
    if len(list(shape)) != 2:
        raise ValueError(
            f'Expected a 2D shape (max_len, features), but got {shape}.')
    max_len, features = shape
    pe = torch.zeros((max_len, features), dtype=dtype).to(device)
    position = torch.arange(0, max_len)[:, None].to(device)
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * torch.exp(torch.arange(0, features // 2) * torch.tensor(scale_factor)).to(device)
    pe[:, :features // 2] = torch.sin(position * div_term).to(device)
    pe[:, features // 2:2 * (features // 2)] = torch.cos(position * div_term).to(device)
    
    return pe

class MlpBlock(nn.Module):
    def __init__(self,
                 emb_dim=512,
                 intermediate_dim=2048,
                 activations=('relu',),
                 kernel_init=nn.init.xavier_uniform_,
                 intermediate_dropout_rate=0.1,
                 dtype=torch.float32):
        super(MlpBlock, self).__init__()
        self.dtype = dtype
        self.intermediate_layers = nn.ModuleList([nn.Linear(in_features=emb_dim, out_features=intermediate_dim), nn.ReLU()])
        self.dropout = nn.Dropout(intermediate_dropout_rate)
        self.dense_layer = nn.Linear(in_features=intermediate_dim, out_features=emb_dim)

    def forward(self, inputs):
        x = inputs

        for layer in self.intermediate_layers:
            x = layer(x)

        x = self.dropout(x)
        output = self.dense_layer(x)
        return output

class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 features,
                 dtype=torch.float32,
                 attend_dtype=None,
                 embedding_init=nn.init.normal_,
                 one_hot=False):
        super(Embed, self).__init__()
        self.num_embeddings = num_embeddings
        self.dtype = dtype
        self.embedding_init = embedding_init
        self.one_hot = one_hot

        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, features)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_init(self.embedding)

    def forward(self, inputs):
        if self.one_hot:
            iota = torch.arange(self.num_embeddings, dtype=torch.int32).to(device)
            one_hot = (inputs[..., None] == iota).type(self.dtype)
            output = torch.matmul(one_hot, self.embedding)
        else:
            output = self.embedding[inputs]

        return output

class FixedEmbed(nn.Module):
    def __init__(self, features, max_length=2048, dtype=torch.float32):
        super(FixedEmbed, self).__init__()
        self.features = features
        self.max_length = max_length
        self.dtype = dtype
        self.embedding = sinusoidal(shape=(self.max_length, self.features), dtype=self.dtype).to(device)

    def forward(self, inputs, decode=False):
        if decode:
            position_embedder_index = self.variable(
                'cache', 'position_embedder_index',
                lambda: torch.array(-1, dtype=torch.uint32).to(device)
            )
            i = position_embedder_index.value
            position_embedder_index.value = i + 1

            return self.embedding[i:i+1, :]

        return self.embedding[inputs, :]

#Referenced https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/t5/modeling_t5.py#L238
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        #convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states