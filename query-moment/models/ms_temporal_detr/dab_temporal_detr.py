import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os


class DABTemporalDetr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass