import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, repeat, rearrange

class SegFormerX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, ):