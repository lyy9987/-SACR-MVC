"""Explicit confidence interventions. none preserves original tensors."""
import torch
import torch.nn.functional as F
def replace_trust(trust, mode):
    if mode == 'none': return trust
    if mode == 'mean': return trust.mean().expand_as(trust)
    if mode in ('one', 'off'): return torch.ones_like(trust)
    raise ValueError('Unknown confidence intervention: '+mode)
def replace_targets(targets, labels, mode):
    if targets is None: return None
    if mode == 'off': return F.one_hot(labels.long(), num_classes=targets.shape[1]).to(targets)
    return targets
