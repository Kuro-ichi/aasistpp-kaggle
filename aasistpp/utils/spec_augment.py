
import torch
import random

def spec_augment(x, time_mask=30, freq_mask=8, p=0.5):
    """
    x: [B, C, F, T], in-place masking (returns a new tensor).
    """
    if random.random() > p:
        return x
    B, C, F, T = x.shape
    x = x.clone()
    # Frequency masks
    for _ in range(2):
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, max(0, F - f))
        x[:, :, f0:f0+f, :] = 0.0
    # Time masks
    for _ in range(2):
        t = random.randint(0, time_mask)
        t0 = random.randint(0, max(0, T - t))
        x[:, :, :, t0:t0+t] = 0.0
    return x
