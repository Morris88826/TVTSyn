from __future__ import annotations

from typing import Any


def _torch():
    import torch
    return torch


def _linear(x: Any, weight: Any, bias: Any | None = None) -> Any:
    return _torch().nn.functional.linear(x, weight, bias)


def _project_qkv(projected: Any, num_heads: int) -> tuple[Any, Any, Any]:
    B, T, three_hd = projected.shape
    head_dim = three_hd // (3 * num_heads)
    qkv = projected.view(B, T, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def _apply_rope(q: Any, k: Any, offset: Any, inv_freq: Any, time_offsets: Any) -> tuple[Any, Any]:
    torch = _torch()
    D = q.shape[-1]
    ts = offset.to(torch.float32) + time_offsets
    rotr = torch.cos(inv_freq * ts.view(1, -1, 1))
    roti = torch.sin(inv_freq * ts.view(1, -1, 1))
    dims = q.shape[:-1]
    q2 = q.view(*dims, D // 2, 2)
    k2 = k.view(*dims, D // 2, 2)
    qr, qi = q2[..., 0].float(), q2[..., 1].float()
    kr, ki = k2[..., 0].float(), k2[..., 1].float()
    dtype = q.dtype
    qo = torch.stack([(qr * rotr - qi * roti).to(dtype), (qr * roti + qi * rotr).to(dtype)], dim=-1)
    ko = torch.stack([(kr * rotr - ki * roti).to(dtype), (kr * roti + ki * rotr).to(dtype)], dim=-1)
    return qo.view(*dims, D), ko.view(*dims, D)


def _ring_complete(
    k_cache: Any, v_cache: Any, end_offset: Any, k: Any, v: Any,
    write_offsets: Any, cache_positions: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    B, H, T, D = k.shape
    capacity = k_cache.shape[2]
    indexes = (write_offsets + end_offset) % capacity
    scatter_idx = indexes.view(1, 1, T, 1).expand(B, H, T, D)
    next_k_cache = k_cache.scatter(2, scatter_idx, k)
    next_v_cache = v_cache.scatter(2, scatter_idx, v)
    next_end = end_offset + T
    invalid = cache_positions >= next_end
    end_index = next_end % capacity
    delta = cache_positions - end_index
    positions = torch.where(delta <= 0, next_end + delta, next_end + delta - capacity)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)
    return next_k_cache, next_v_cache, next_end, positions


def _functional_mha(
    attn: Any, x: Any, tr: int, k_cache: Any, v_cache: Any, end_offset: Any, constants: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    B, T, _ = x.shape
    Tq = T - int(tr)
    q, k, v = _project_qkv(_linear(x, attn.in_proj_weight, attn.in_proj_bias), int(attn.num_heads))
    if getattr(attn, "rope", None) is not None:
        q, k = _apply_rope(q, k, end_offset, constants["rope_inv_freq"], constants["query_offsets_float"])
    next_k_cache, next_v_cache, next_end, pos_k_cache = _ring_complete(
        k_cache, v_cache, end_offset, k[:, :, :Tq, :], v[:, :, :Tq, :],
        constants["write_offsets"], constants["cache_positions"],
    )
    if tr > 0:
        k_all = torch.cat([next_k_cache, k[:, :, Tq:, :]], dim=2)
        v_all = torch.cat([next_v_cache, v[:, :, Tq:, :]], dim=2)
        pos_k = torch.cat([pos_k_cache, end_offset + constants["future_offsets"]], dim=0)
    else:
        k_all, v_all, pos_k = next_k_cache, next_v_cache, pos_k_cache
    if bool(getattr(attn, "causal", False)):
        pos_k_2d = pos_k.view(1, -1)
        pos_q = end_offset + constants["query_offsets"].view(-1, 1)
        delta = pos_q - pos_k_2d
        attn_bias = pos_k_2d >= 0
        context = getattr(attn, "context", None)
        if context is not None:
            attn_bias = attn_bias & (delta < int(context))
        attn_bias = attn_bias & (delta >= -int(getattr(attn, "right_context", 0) or 0))
    else:
        attn_bias = None
    y = torch.nn.functional.scaled_dot_product_attention(q, k_all, v_all, attn_bias, dropout_p=0.0)
    y = y.permute(0, 2, 1, 3).contiguous().view(B, T, int(attn.embed_dim))
    return _linear(y, attn.out_proj.weight, attn.out_proj.bias), next_k_cache, next_v_cache, next_end


def functional_transformer(transformer: Any, x: Any, tr: int, state_tensors: tuple[Any, ...], next_state: list[Any], layout: dict[str, Any]) -> Any:
    torch = _torch()
    B, T, C = x.shape
    transformer_offset_idx = layout.get("offset_idx")
    transformer_offset = state_tensors[transformer_offset_idx] if transformer_offset_idx is not None else torch.zeros(1, dtype=torch.long, device=x.device)
    if getattr(transformer, "positional_embedding", "none") in {"sin", "sin_rope"}:
        from tvtsyn.components.transformer import create_sin_embedding
        positions = layout["position_offsets"].view(1, -1, 1) + transformer_offset.view(-1, 1, 1)
        x = x + float(getattr(transformer, "positional_scale", 1.0)) * create_sin_embedding(
            positions, C, max_period=float(getattr(transformer, "max_period", 10000)), dtype=x.dtype
        )
    for layer, layer_layout in zip(transformer.layers, layout["layers"]):
        if not getattr(layer, "skip_self_attn", False):
            residual = x
            y, next_k, next_v, next_end = _functional_mha(
                layer.self_attn, layer.norm1(x), tr,
                state_tensors[layer_layout["k_idx"]], state_tensors[layer_layout["v_idx"]],
                state_tensors[layer_layout["end_idx"]], layer_layout["constants"],
            )
            next_state[layer_layout["k_idx"]] = next_k
            next_state[layer_layout["v_idx"]] = next_v
            next_state[layer_layout["end_idx"]] = next_end
            x = residual + layer.layer_scale_1(y)
        residual = x
        x_norm = layer.norm2(x)
        update = layer.linear2(layer.activation(layer.linear1(x_norm))) if getattr(layer, "gating", None) is None else layer.gating(x_norm)
        x = residual + layer.layer_scale_2(update)
        layer_offset_idx = layer_layout.get("layer_offset_idx")
        if layer_offset_idx is not None:
            next_state[layer_offset_idx] = state_tensors[layer_offset_idx] + T
    if transformer_offset_idx is not None:
        next_state[transformer_offset_idx] = transformer_offset + T
    return x


def functional_projected_transformer(projected: Any, x: Any, tr: int, state_tensors: tuple[Any, ...], next_state: list[Any], layout: dict[str, Any]) -> list[Any]:
    if getattr(projected, "conv_layout", False):
        x = x.transpose(1, 2)
    if getattr(projected, "input_proj", None) is not None:
        x = projected.input_proj(x)
    z = functional_transformer(projected.transformer, x, tr, state_tensors, next_state, layout)
    ys = []
    for output_proj in projected.output_projs:
        y = output_proj(z)
        if getattr(projected, "conv_layout", False):
            y = y.transpose(1, 2)
        ys.append(y)
    return ys
