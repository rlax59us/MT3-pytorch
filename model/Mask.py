"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
from typing import Callable

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_attention_mask(query_input, key_input, pairwise_fn: Callable=torch.mul, extra_batch_dims=0, dtype=torch.float32):
    mask = pairwise_fn(
        query_input.unsqueeze(-1),
        key_input.unsqueeze(-2)
    )
    mask = mask.unsqueeze(-3)
    
    for i in range(extra_batch_dims):
        mask = mask.unsqueeze(i)
    
    return mask.type(dtype)

def make_causal_mask(x, extra_batch_dims=0, dtype=torch.float32):
    idxs = torch.arange(x.shape[-1], dtype=torch.int32).expand(x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        torch.greater_equal,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype)

def combine_masks(*masks, dtype=torch.float32):
    masks = [m.to(device) for m in masks if m is not None]
    
    if not masks:
        return None
    
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    
    mask, *other_masks = masks
    
    for other_mask in other_masks:
        mask = torch.logical_and(mask, other_mask)

    return mask.type(dtype)

def combine_biases(*masks):
    masks = [m for m in masks if m is not None]
    
    if not masks:
        return None
    
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    
    mask, *other_masks = masks
    
    for other_mask in other_masks:
        mask = mask + other_mask
    
    return mask

def make_decoder_mask(decoder_target_tokens,
                      dtype,
                      decoder_causal_attention=None,
                      decoder_segment_ids=None):
    masks = []
    
    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

    if decoder_causal_attention is not None:
        inputs_mask = make_attention_mask(
            decoder_causal_attention,
            decoder_causal_attention,
            torch.logical_and,
            dtype=dtype)
        masks.append(torch.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
        masks.append(causal_mask)

    masks.append(
        make_attention_mask(
            decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

    # Packing mask
    if decoder_segment_ids is not None:
        masks.append(
            make_attention_mask(
                decoder_segment_ids, decoder_segment_ids, torch.equal, dtype=dtype))

    return combine_masks(*masks, dtype=dtype)