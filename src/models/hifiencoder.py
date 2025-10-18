import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

from src.components.causal_conv import CausalConv
from src.components.causal_resblock import ResBlock
from src.components.fast_glu import FastGLU
from src.components.lookahead_block import LookaheadBlock
from src.components.streaming import StreamingContainer, StreamingAdd
from src.utils.compile import torch_compile_lazy

LRELU_SLOPE = 0.1


class HiFiEncoder(StreamingContainer):
    def __init__(self, h):
        super(HiFiEncoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.up_buf_len = 1

        resblock = ResBlock
        
        self.conv_pre = CausalConv(1, h.downsample_initial_channel, 7, 1)

        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, h.downsample_kernel_sizes)):
            self.downs.append(
                CausalConv(h.downsample_initial_channel * (2 ** i), 
                           h.downsample_initial_channel * (2 ** (i + 1)), 
                           k, u))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.downsample_initial_channel * (2 ** i)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, 0, d))

        self.conv_post = CausalConv(ch * 2, h.upsample_initial_channel, 7, 1)
        self.final_act = FastGLU(h.upsample_initial_channel)

        if h.lookahead > 0:
            self.lookahead = LookaheadBlock(h.upsample_initial_channel, h.lookahead)
    
        self.code_predictor = nn.Linear(h.upsample_initial_channel, h.num_codes)
        self.add_list = nn.ModuleList()
        for i in range(self.num_downsamples):
            for j in range(self.num_kernels):
                self.add_list.append(StreamingAdd())
    
    # @torch_compile_lazy
    def forward(self, x):

        #pre conv buff
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs_ = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.add_list[i * self.num_kernels + j](xs, xs_)
                    # xs += xs_

            x = xs / self.num_kernels

            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

        x = F.leaky_relu(x)

        x = self.conv_post(x)
        # print("post conv shape", x.shape)
        z_embed = self.final_act(x.transpose(1, 2)).transpose(1, 2)

        # print("z_embed shape", z_embed.shape)
        if hasattr(self, 'lookahead'):
            z_embed = self.lookahead(z_embed)
            # print("lookahead z_embed shape", z_embed.shape)


        code = self.code_predictor(z_embed.transpose(1, 2)).transpose(1, 2)
        # print("code shape", code.shape)
        
        return code, z_embed