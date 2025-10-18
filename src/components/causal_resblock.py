import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from src.components.streaming import StreamingContainer, StreamingAdd
from src.components.causal_conv import CausalConv
from src.utils.compile import torch_compile_lazy

LRELU_SLOPE = 0.1


class ResBlock(StreamingContainer):
    def __init__(self, h, channels, kernel_size=3, padding=0, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h

        self.channels = channels
        self.kernel_size = kernel_size
        num_layers = 3

        self.convs1 = nn.ModuleList([
            CausalConv(channels, channels, kernel_size=kernel_size, dilation=dilation[0], pad_mode="constant", causal=True),
            CausalConv(channels, channels, kernel_size=kernel_size, dilation=dilation[1], pad_mode="constant", causal=True),
            CausalConv(channels, channels, kernel_size=kernel_size, dilation=dilation[2], pad_mode="constant", causal=True)
        ])

        self.convs2 = nn.ModuleList([
            CausalConv(channels, channels, kernel_size=kernel_size, dilation=1, pad_mode="constant", causal=True),
            CausalConv(channels, channels, kernel_size=kernel_size, dilation=1, pad_mode="constant", causal=True),
            CausalConv(channels, channels, kernel_size=kernel_size, dilation=1, pad_mode="constant", causal=True)
        ])

        self.streaming_add = StreamingAdd()

    
    def forward(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):

            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)

            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)

            # Residual connection
            x = self.streaming_add(xt, x)
        return x