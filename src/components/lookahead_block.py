import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from src.components.streaming import StreamingContainer
from src.components.causal_conv import CausalConv


class LookaheadBlock(StreamingContainer):
    def __init__(self, in_channel, kernel_size):
        super(LookaheadBlock, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.layer = CausalConv(
            in_channel, in_channel, 
            kernel_size=kernel_size, 
            dilation=1, 
            causal=False, 
            pad_mode="constant"
        )
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(in_channel)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x