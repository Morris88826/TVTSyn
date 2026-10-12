from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for tvtsyn.streaming. Install project requirements "
            "or run these tests in the TVTSyn training environment."
        ) from exc
    return torch


@dataclass(frozen=True)
class TVTSynStateEntry:
    name: str
    shape: tuple[int, ...]
    dtype: object
    device: object


@dataclass(frozen=True)
class TVTSynStateSchema:
    entries: tuple[TVTSynStateEntry, ...]
    batch_size: int
    chunk_samples: int
    chunk_frames: int
    sample_rate: int
    frame_rate: int
    hop_length: int
    encoder_dim: int
    speaker_dim: int
    lookahead_frames: int

    @property
    def state_names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    def index(self, name: str) -> int:
        return self.state_names.index(name)

    def to_jsonable(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "chunk_samples": self.chunk_samples,
            "chunk_frames": self.chunk_frames,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "hop_length": self.hop_length,
            "encoder_dim": self.encoder_dim,
            "speaker_dim": self.speaker_dim,
            "lookahead_frames": self.lookahead_frames,
            "entries": [
                {
                    "index": idx,
                    "name": entry.name,
                    "shape": list(entry.shape),
                    "dtype": str(entry.dtype).replace("torch.", ""),
                    "device": str(entry.device),
                }
                for idx, entry in enumerate(self.entries)
            ],
        }


def _get_attr(obj: object, dotted: str, default: object = None) -> object:
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _model_meta(model: object, chunk_samples: int) -> dict[str, int]:
    wrapped = getattr(model, "model", model)
    sample_rate = int(getattr(wrapped, "hparams", {}).get("sample_rate", 16000)) if hasattr(wrapped, "hparams") else 16000
    frame_rate = int(getattr(wrapped, "hparams", {}).get("frame_rate", 50)) if hasattr(wrapped, "hparams") else 50
    hop_length = int(sample_rate // frame_rate)
    encoder_dim = int(getattr(wrapped, "hparams", {}).get("encoder_dim", 512)) if hasattr(wrapped, "hparams") else 512
    decoder = _get_attr(wrapped, "decoder.decoder_transformer")
    if decoder is not None:
        encoder_dim = int(getattr(decoder, "input_dimension", encoder_dim))
        speaker_dim = int(getattr(decoder, "timbre_encoder.gtm.net.0.in_features", 704))
    else:
        speaker_dim = 704
    enc_tr = _get_attr(wrapped, "encoder_net.encoder_transformer.transformer")
    if enc_tr is None:
        enc_tr = _get_attr(wrapped, "encoder.net.encoder_transformer.transformer")
    lookahead = 0
    if enc_tr is not None and hasattr(enc_tr, "layers") and len(enc_tr.layers) > 0:
        lookahead = int(getattr(enc_tr.layers[0].self_attn, "right_context", 0))
    else:
        lookahead = int(getattr(_get_attr(wrapped, "encoder.net.encoder_transformer"), "right_context", 0) or 0)
    chunk_frames = max(1, chunk_samples // hop_length)
    return {
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "hop_length": hop_length,
        "encoder_dim": encoder_dim,
        "speaker_dim": speaker_dim,
        "lookahead_frames": lookahead,
        "chunk_frames": chunk_frames,
    }


def _transformer_entries(prefix: str, transformer: object, batch_size: int, dtype: object, device: object) -> list[TVTSynStateEntry]:
    entries: list[TVTSynStateEntry] = []
    layers = list(getattr(transformer, "layers", []) or [])
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        context = int(getattr(attn, "context", 0) or 0)
        num_heads = int(getattr(attn, "num_heads", 1))
        embed_dim = int(getattr(attn, "embed_dim", 1))
        head_dim = embed_dim // num_heads
        capacity = max(1, context)
        entries.extend(
            [
                TVTSynStateEntry(f"{prefix}.layers.{layer_idx}.self_attn.k_cache", (batch_size, num_heads, capacity, head_dim), dtype, device),
                TVTSynStateEntry(f"{prefix}.layers.{layer_idx}.self_attn.v_cache", (batch_size, num_heads, capacity, head_dim), dtype, device),
                TVTSynStateEntry(f"{prefix}.layers.{layer_idx}.self_attn.end_offset", (1,), _torch().long, device),
                TVTSynStateEntry(f"{prefix}.layers.{layer_idx}.offset", (1,), _torch().long, device),
            ]
        )
    entries.append(TVTSynStateEntry(f"{prefix}.offset", (1,), _torch().long, device))
    return entries


def _streaming_module_entries(model: object, batch_size: int, dtype: object, device: object) -> list[TVTSynStateEntry]:
    torch = _torch()
    entries: list[TVTSynStateEntry] = []
    try:
        from tvtsyn.components.conv import StreamingConv1d
        from tvtsyn.components.streaming import RawStreamingConv1d, RawStreamingConvTranspose1d, StreamingAdd
        from tvtsyn.components.tv_timbre import StreamingGlobalOnlyTVTimbre
    except Exception:
        return entries

    for name, module in getattr(model, "named_modules", lambda: [])():
        if not name:
            continue
        if isinstance(module, RawStreamingConv1d):
            kernel = int((module.kernel_size[0] - 1) * module.dilation[0] + 1)
            cache_len = max(0, kernel - int(module.stride[0]))
            entries.append(TVTSynStateEntry(f"{name}.previous", (batch_size, module.in_channels, cache_len), dtype, device))
            entries.append(TVTSynStateEntry(f"{name}.valid", (1,), torch.long, device))
        elif isinstance(module, RawStreamingConvTranspose1d):
            partial_len = max(1, int(module.kernel_size[0] - module.stride[0]))
            entries.append(TVTSynStateEntry(f"{name}.partial", (batch_size, module.out_channels, partial_len), dtype, device))
            entries.append(TVTSynStateEntry(f"{name}.valid", (1,), torch.long, device))
        elif isinstance(module, StreamingConv1d):
            entries.append(TVTSynStateEntry(f"{name}.padding_to_add", (1,), torch.long, device))
        elif isinstance(module, StreamingAdd):
            # TODO: confirm exact per-add channel/length bounds from paired branches before strict parity export.
            entries.append(TVTSynStateEntry(f"{name}.previous_x", (batch_size, 1, 1), dtype, device))
            entries.append(TVTSynStateEntry(f"{name}.previous_y", (batch_size, 1, 1), dtype, device))
            entries.append(TVTSynStateEntry(f"{name}.valid", (1,), torch.long, device))
        elif isinstance(module, StreamingGlobalOnlyTVTimbre):
            entries.append(TVTSynStateEntry(f"{name}.pos_offset", (1,), torch.long, device))
    return entries


def make_tvtsyn_state_schema(
    model: object,
    batch_size: int,
    chunk_samples: int,
    device: object,
    dtype: object,
) -> TVTSynStateSchema:
    torch = _torch()
    if chunk_samples not in (320, 640, 960, 1280, 1600, 2240):
        raise ValueError(f"Task 0 supports fixed chunk sizes 320, 640, 960, 1280, 1600, and 2240 samples; got {chunk_samples}.")

    meta = _model_meta(model, chunk_samples)
    entries: list[TVTSynStateEntry] = [
        TVTSynStateEntry("control.frames_seen", (1,), torch.long, device),
        TVTSynStateEntry("control.ready", (1,), torch.long, device),
        TVTSynStateEntry("encoder.lookahead_fifo", (batch_size, meta["encoder_dim"], (2 * meta["chunk_frames"]) + meta["lookahead_frames"]), dtype, device),
        TVTSynStateEntry("encoder.encoded_frames_seen", (1,), torch.long, device),
    ]

    wrapped = getattr(model, "model", model)
    enc_transformer = _get_attr(wrapped, "encoder_net.encoder_transformer.transformer")
    if enc_transformer is None:
        enc_transformer = _get_attr(wrapped, "encoder.net.encoder_transformer.transformer")
    if enc_transformer is not None:
        entries.extend(_transformer_entries("encoder.transformer", enc_transformer, batch_size, dtype, device))
    dec_transformer = _get_attr(wrapped, "decoder.decoder_transformer")
    if dec_transformer is not None:
        for block_name in ("b1_transformer", "b2_transformer", "b3_transformer"):
            block = getattr(dec_transformer, block_name, None)
            if block is not None:
                entries.extend(_transformer_entries(f"decoder.{block_name}", block, batch_size, dtype, device))
    entries.extend(_streaming_module_entries(wrapped, batch_size, dtype, device))

    seen: set[str] = set()
    deduped: list[TVTSynStateEntry] = []
    for entry in entries:
        if entry.name in seen:
            continue
        seen.add(entry.name)
        deduped.append(entry)

    return TVTSynStateSchema(
        entries=tuple(deduped),
        batch_size=batch_size,
        chunk_samples=chunk_samples,
        chunk_frames=meta["chunk_frames"],
        sample_rate=meta["sample_rate"],
        frame_rate=meta["frame_rate"],
        hop_length=meta["hop_length"],
        encoder_dim=meta["encoder_dim"],
        speaker_dim=meta["speaker_dim"],
        lookahead_frames=meta["lookahead_frames"],
    )


def init_tvtsyn_state(
    model: object,
    batch_size: int,
    chunk_samples: int,
    device: object,
    dtype: object,
) -> tuple[object, ...]:
    torch = _torch()
    schema = make_tvtsyn_state_schema(model, batch_size, chunk_samples, device, dtype)
    tensors = []
    for entry in schema.entries:
        tensors.append(torch.zeros(entry.shape, device=entry.device, dtype=entry.dtype))
    return tuple(tensors)


def describe_state(state: Sequence[object], schema: TVTSynStateSchema | None = None) -> list[dict]:
    rows: list[dict] = []
    names = schema.state_names if schema is not None else [f"state_{idx:03d}" for idx in range(len(state))]
    for idx, tensor in enumerate(state):
        rows.append(
            {
                "index": idx,
                "name": names[idx] if idx < len(names) else f"state_{idx:03d}",
                "shape": list(getattr(tensor, "shape", ())),
                "dtype": str(getattr(tensor, "dtype", "")),
                "device": str(getattr(tensor, "device", "")),
            }
        )
    return rows


def validate_state(state: Sequence[object], schema: TVTSynStateSchema) -> None:
    torch = _torch()
    if len(state) != len(schema.entries):
        raise ValueError(f"State length mismatch: got {len(state)}, expected {len(schema.entries)}.")
    for idx, (tensor, entry) in enumerate(zip(state, schema.entries)):
        if not torch.is_tensor(tensor):
            raise TypeError(f"state[{idx}] {entry.name} is not a tensor: {type(tensor)!r}.")
        if tuple(tensor.shape) != entry.shape:
            raise ValueError(f"state[{idx}] {entry.name} shape mismatch: got {tuple(tensor.shape)}, expected {entry.shape}.")
        if tensor.dtype != entry.dtype:
            raise TypeError(f"state[{idx}] {entry.name} dtype mismatch: got {tensor.dtype}, expected {entry.dtype}.")
        if str(tensor.device) != str(entry.device):
            raise ValueError(f"state[{idx}] {entry.name} device mismatch: got {tensor.device}, expected {entry.device}.")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"state[{idx}] {entry.name} contains NaN or Inf.")
        if entry.name.endswith(("end_offset", "offset", "frames_seen", "encoded_frames_seen", "lookahead_count", "pos_offset")):
            if (tensor < 0).any():
                raise ValueError(f"state[{idx}] {entry.name} contains a negative counter.")
        if entry.name.endswith(("k_cache", "v_cache")) and tensor.shape[2] != entry.shape[2]:
            raise ValueError(f"state[{idx}] {entry.name} cache length grew from {entry.shape[2]} to {tensor.shape[2]}.")


def state_size_bytes(state: Iterable[object]) -> int:
    return sum(int(t.numel() * t.element_size()) for t in state if hasattr(t, "numel") and hasattr(t, "element_size"))
