from dataclasses import dataclass
import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from src.components.conv import (
    get_extra_padding_for_conv1d, 
    pad1d,
    unpad1d
)
from src.components.streaming import (
    RawStreamingConv1d, 
    RawStreamingConvTranspose1d, 
    StreamingModule
)
from src.utils.compile import torch_compile_lazy



@dataclass
class _StreamingConv1dState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self):
        self.padding_to_add = self.original_padding_to_add



class CausalConv(StreamingModule[_StreamingConv1dState]):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, dilation=1, causal=True, pad_mode="constant"):
        super(CausalConv, self).__init__()
        self.in_channel = in_channel
        self.buf_len = (kernel_size - 1) * dilation
        self.stride = stride
        self.causal = causal
        self.pad_mode = pad_mode
        self.layer = weight_norm(RawStreamingConv1d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation))

    @property
    def _stride(self) -> int:
        return self.layer.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.layer.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        dilation = self.layer.dilation[0]
        return (
            self._kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
        # assert self.causal, "streaming is only supported for causal convs"
        # return _StreamingConv1dState(self._padding_total, self._padding_total)
        if self.causal:
            return _StreamingConv1dState(self._padding_total, self._padding_total)
        else:
            return _StreamingConv1dState(
                self._padding_total // 2, self._padding_total // 2
            )

    def forward(self, x):
        B, C, T = x.shape
        padding_total = self._padding_total
        extra_padding = get_extra_padding_for_conv1d(
            x, self._effective_kernel_size, self._stride, padding_total
        )
        state = self._streaming_state
        if state is None:
            if self.causal:
                # Left padding for causal
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        return self.layer(x)
    

@dataclass
class _StreamingConvTr1dState:
    pass

    def reset(self):
        pass


class CausalConvTranspose1D(StreamingModule[_StreamingConvTr1dState]):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, causal=True, trim_right_ratio=1.0):
        """
        Causal ConvTranspose1D layer with streaming support.
        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to both sides of the input.
            causal (bool): If True, use causal convolution.
            trim_right_ratio (float): Ratio to trim the right side of the output.
        """
        super(CausalConvTranspose1D, self).__init__()
        self.in_channel = in_channel
        self.buf_len = kernel_size - 1
        self.upsample_factor = stride
        self.layer = weight_norm(RawStreamingConvTranspose1d(in_channel, out_channel, kernel_size, stride, padding))

        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0
    
    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTr1dState:
        assert self.causal, "streaming is only supported for causal convtrs"
        return _StreamingConvTr1dState()

    def forward(self, x):
        kernel_size = self.layer.kernel_size[0]
        stride = self.layer.stride[0]
        padding_total = kernel_size - stride

        y = self.layer(x)

        if not self.is_streaming:
            # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
            # removed at the very end, when keeping only the right length for the output,
            # as removing it here would require also passing the length at the matching layer
            # in the encoder.
            if self.causal:
                # Trim the padding on the right according to the specified ratio
                # if trim_right_ratio = 1.0, trim everything from right
                padding_right = math.ceil(padding_total * self.trim_right_ratio)
                padding_left = padding_total - padding_right
                y = unpad1d(y, (padding_left, padding_right))
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                y = unpad1d(y, (padding_left, padding_right))
        return y