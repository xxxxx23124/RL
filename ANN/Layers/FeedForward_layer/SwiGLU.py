import torch
import torch.nn as nn
import torch.nn.functional as F
from FeedForwardConfig import FeedForwardConfig

class SwiGLUFeedForward(nn.Module):
    def __init__(self, args: FeedForwardConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.w1 = nn.Linear(args.d_model, args.d_inner, device=device)
        self.w2 = nn.Linear(args.d_inner, args.d_model, device=device)
        self.w3 = nn.Linear(args.d_model, args.d_inner, device=device)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))