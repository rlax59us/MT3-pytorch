"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Layers import *
from model.Mask import *
from data.constants import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)
        
    def encode(
            self, 
            encoder_input_tokens, 
            encoder_segment_ids=None, 
            enable_dropout=True
            ):
        assert encoder_input_tokens.ndim == 3  # (batch, length, depth)

        encoder_mask = make_attention_mask(
            torch.ones(encoder_input_tokens.shape[:-1]),
            torch.ones(encoder_input_tokens.shape[:-1]),
            dtype=self.config.dtype
        )

        if encoder_segment_ids is not None:
            encoder_mask = combine_masks(
                encoder_mask,
                make_attention_mask(
                    encoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype
                )
            )

        return self.encoder(encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)
    
    def decode(
            self, 
            encoded, 
            encoder_input_tokens, 
            decoder_input_tokens, 
            decoder_target_tokens, 
            encoder_segment_ids=None, 
            decoder_segment_ids=None, 
            decoder_positions=None, 
            enable_dropout=True, 
            decode=False, #decode: Whether to prepare and use an autoregressive cache
            max_decode_length=None
            ):

        if decode:
            decoder_mask = None
            encoder_decoder_mask = make_attention_mask(
                torch.ones_like(decoder_target_tokens).to(device),
                torch.ones(encoder_input_tokens.shape[:-1]).to(device),
                dtype=self.config.dtype
            )
        else:
            decoder_mask = make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=self.config.dtype,
                decoder_segment_ids=decoder_segment_ids
            )
            encoder_decoder_mask = make_attention_mask(
                decoder_target_tokens > 0,
                torch.ones(encoder_input_tokens.shape[:-1]).to(device),
                dtype=self.config.dtype
            )

        if encoder_segment_ids is not None:
            if decode:
                raise ValueError('During decoding, packing should not be used but `encoder_segment_ids` was passed to `Transformer.decode`.')

            encoder_decoder_mask = combine_masks(
                encoder_decoder_mask,
                make_attention_mask(
                    decoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype
                )
            )

        logits = self.decoder(
            encoded,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length
            )
        return logits.type(self.config.dtype)
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = TOKEN_START

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        return shifted_input_ids.to(device)
    
    def forward(self, encoder_input_tokens, decoder_target_tokens, decoder_input_tokens=None, encoder_segment_ids=None, decoder_segment_ids=None, encoder_positions=None, decoder_positions=None, enable_dropout=True, decode=False):
        if decoder_input_tokens == None:
            decoder_input_tokens = self._shift_right(decoder_target_tokens)
        encoded = self.encode(encoder_input_tokens, encoder_segment_ids=encoder_segment_ids, enable_dropout=enable_dropout)
        return self.decode(encoded, encoder_input_tokens, decoder_input_tokens, decoder_target_tokens, encoder_segment_ids=encoder_segment_ids, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions, enable_dropout=enable_dropout, decode=decode)

    def generate(self, primer=None, target_seq_length=1024):
        num_primer = len(primer)
        len_primer = len(primer[0])
        gen_tokens = torch.LongTensor([self.pad_token for i in range(target_seq_length-len_primer)]).expand(num_primer, target_seq_length-len_primer)
        gen_tokens = torch.concat((primer.type(torch.long).to(device), gen_tokens.to(device)), dim=-1).to(device)

        i = num_primer
        while (i < target_seq_length):
            logits, _ = self.forward(gen_tokens[..., :i])
            probs = self.softmax(logits)[..., :self.eos_token]
            token_probs = probs[:, i - 1, :]

            next_token = torch.argmax(token_probs)
            gen_tokens[:, i] = next_token

            if next_token == self.eos_token:
                break
            i += 1

        return gen_tokens[:, :i]