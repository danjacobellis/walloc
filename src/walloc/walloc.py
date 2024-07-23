import torch
import torch.nn as nn

class NormClip(nn.Module):
    def __init__(self, min_val, max_val):
        super(NormClip, self).__init__()
        self.max_val = max_val
        self.hardtanh = nn.Hardtanh(min_val=min_val, max_val=max_val)
    def forward(self, x):
        norm = x.norm(dim=1, keepdim=True)
        direction = x / (norm + 1e-8)  # Add epsilon to avoid division by zero
        clipped_norm = self.hardtanh(norm)
        return direction * clipped_norm