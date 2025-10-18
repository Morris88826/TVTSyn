import torch
import torch.nn.functional as F
import torch.nn as nn

from src.components.causal_conv import CausalConv, CausalConvTranspose1D
from src.components.causal_resblock import ResBlock
from src.components.speaker_adapter import SpeakerAdapter

from src.components.streaming import StreamingContainer, StreamingAdd
from src.utils.compile import torch_compile_lazy


LRELU_SLOPE = 0.1


class Generator(StreamingContainer):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.up_buf_len = 1

        resblock = ResBlock
        
        self.conv_pre = CausalConv(getattr(h, "model_in_dim", 128), h.upsample_initial_channel, 7, 1)

        self.ups = nn.ModuleList()
        self.spk_adapters = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                CausalConvTranspose1D(h.upsample_initial_channel // (2 ** i), 
                                      h.upsample_initial_channel // (2 ** (i + 1)), 
                                      k, u))
            if i!=0:
                self.spk_adapters.append(SpeakerAdapter(h.spkr_embed_dim, h.upsample_initial_channel // (2 ** i)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, 0, d))

        self.add_list = nn.ModuleList()
        for i in range(self.num_upsamples):
            for j in range(self.num_kernels):
                self.add_list.append(StreamingAdd())

        self.conv_post = CausalConv(ch, 1, 7, 1)
        # for l in self.ups:
        #     l.init_weights()
        # self.conv_post.layer.apply(init_weights)

    # @torch_compile_lazy
    def forward(self, x, embeds):
        #Add post conv buff
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            if i!=0:
                x = self.spk_adapters[i-1](x, embeds)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs_ = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.add_list[i * self.num_kernels + j](xs, xs_)
                    # xs += xs_
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        #Add post conv buff
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x