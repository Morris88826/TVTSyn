from __future__ import annotations

from typing import Any


def _torch():
    import torch

    return torch


def _state_index(state_names: list[str], name: str) -> int | None:
    try:
        return state_names.index(name)
    except ValueError:
        return None


def _set_scalar(next_state: list[Any], idx: int | None, value: Any) -> None:
    if idx is not None:
        next_state[idx] = value.reshape_as(next_state[idx]) if hasattr(value, "reshape_as") else value


def functional_streaming_conv1d(
    module: Any,
    x: Any,
    name: str,
    state_names: list[str],
    state_tensors: tuple[Any, ...],
    next_state: list[Any],
) -> Any:
    """Tensor-only equivalent of causal StreamingConv1d for fixed chunks.

    The TVTSyn configs use causal, constant-padded convolutions. For that case
    the initial left pad and all later overlap are represented by a fixed
    previous-input tensor of length `effective_kernel - stride`.
    """

    torch = _torch()
    prev_idx = _state_index(state_names, f"{name}.conv.conv.previous")
    valid_idx = _state_index(state_names, f"{name}.conv.conv.valid")
    pad_idx = _state_index(state_names, f"{name}.padding_to_add")
    if prev_idx is None:
        return module(x)

    prev = state_tensors[prev_idx]
    prev_len = int(prev.shape[-1])
    if prev_len > 0:
        x_in = torch.cat([prev, x], dim=-1)
    else:
        x_in = x
    y = module.conv(x_in)

    if prev_len > 0:
        next_state[prev_idx] = x_in[..., -prev_len:]
    else:
        next_state[prev_idx] = prev
    _set_scalar(next_state, valid_idx, torch.ones_like(state_tensors[valid_idx]) if valid_idx is not None else None)
    _set_scalar(next_state, pad_idx, torch.zeros_like(state_tensors[pad_idx]) if pad_idx is not None else None)
    return y


def functional_streaming_conv_transpose1d(
    module: Any,
    x: Any,
    name: str,
    state_names: list[str],
    state_tensors: tuple[Any, ...],
    next_state: list[Any],
) -> Any:
    """Tensor-only equivalent of causal StreamingConvTranspose1d."""

    torch = _torch()
    partial_idx = _state_index(state_names, f"{name}.convtr.convtr.partial")
    valid_idx = _state_index(state_names, f"{name}.convtr.convtr.valid")
    if partial_idx is None:
        return module(x)

    out = module.convtr(x)
    partial = state_tensors[partial_idx]
    partial_len = int(partial.shape[-1])
    if partial_len == 0:
        return out

    valid = state_tensors[valid_idx].to(dtype=out.dtype).view(1, 1, 1) if valid_idx is not None else out.new_ones(1, 1, 1)
    correction = partial
    bias = getattr(module.convtr.convtr, "bias", None)
    if bias is not None:
        correction = correction - bias.view(1, -1, 1)
    prefix = out[..., :partial_len] + valid * correction
    out = torch.cat([prefix, out[..., partial_len:]], dim=-1)
    next_state[partial_idx] = out[..., -partial_len:]
    _set_scalar(next_state, valid_idx, torch.ones_like(state_tensors[valid_idx]) if valid_idx is not None else None)
    return out[..., :-partial_len]


def functional_streaming_add(
    x: Any,
    y: Any,
    name: str,
    state_names: list[str],
    state_tensors: tuple[Any, ...],
    next_state: list[Any],
) -> Any:
    """Tensorized StreamingAdd.

    In the current causal TVTSyn configs the paired branches have equal length.
    The crop keeps the operation robust if a branch is one frame shorter.
    """

    torch = _torch()
    m_l = min(int(x.shape[-1]), int(y.shape[-1]))
    px_idx = _state_index(state_names, f"{name}.previous_x")
    py_idx = _state_index(state_names, f"{name}.previous_y")
    valid_idx = _state_index(state_names, f"{name}.valid")
    if px_idx is not None:
        next_state[px_idx] = torch.zeros_like(state_tensors[px_idx])
    if py_idx is not None:
        next_state[py_idx] = torch.zeros_like(state_tensors[py_idx])
    _set_scalar(next_state, valid_idx, torch.ones_like(state_tensors[valid_idx]) if valid_idx is not None else None)
    return x[..., :m_l] + y[..., :m_l]


def functional_module(
    module: Any,
    x: Any,
    name: str,
    state_names: list[str],
    state_tensors: tuple[Any, ...],
    next_state: list[Any],
) -> Any:
    from tvtsyn.components.conv import StreamingConv1d, StreamingConvTranspose1d, TransposedLayerNorm
    from tvtsyn.components.convnext import ConvNeXtBlock
    from tvtsyn.components.seanet import SEANetResnetBlock

    if module.__class__.__name__ in {"SEANetEncoder", "SEANetDecoder", "ConditionalSEANetDecoder"} and hasattr(module, "model"):
        return functional_module(module.model, x, f"{name}.model", state_names, state_tensors, next_state)
    if isinstance(module, StreamingConv1d):
        return functional_streaming_conv1d(module, x, name, state_names, state_tensors, next_state)
    if isinstance(module, StreamingConvTranspose1d):
        return functional_streaming_conv_transpose1d(module, x, name, state_names, state_tensors, next_state)
    if isinstance(module, SEANetResnetBlock):
        u = x if module.shortcut.__class__.__name__ == "Identity" else functional_module(
            module.shortcut, x, f"{name}.shortcut", state_names, state_tensors, next_state
        )
        v = functional_module(module.block, x, f"{name}.block", state_names, state_tensors, next_state)
        return functional_streaming_add(u, v, f"{name}.add", state_names, state_tensors, next_state)
    if isinstance(module, ConvNeXtBlock):
        residual = x
        y = functional_module(module.dwconv, x, f"{name}.dwconv", state_names, state_tensors, next_state)
        y = module.norm(y)
        y = y.permute(0, 2, 1)
        y = module.pwconv1(y)
        y = module.act(y)
        y = module.pwconv2(y)
        if module.gamma is not None:
            y = module.gamma * y
        y = y.permute(0, 2, 1)
        y = module.drop_path(y)
        return functional_streaming_add(residual, y, f"{name}.add", state_names, state_tensors, next_state)
    if hasattr(module, "__iter__") and module.__class__.__name__ in {"Sequential", "ModuleList"}:
        y = x
        for idx, child in enumerate(module):
            y = functional_module(child, y, f"{name}.{idx}" if name else str(idx), state_names, state_tensors, next_state)
        return y
    if isinstance(module, TransposedLayerNorm):
        return module(x)
    return module(x)
