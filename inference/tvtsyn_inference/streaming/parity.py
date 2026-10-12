from __future__ import annotations

from dataclasses import dataclass


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for parity helpers.") from exc
    return torch


@dataclass(frozen=True)
class ParityMetrics:
    max_abs: float
    mean_abs: float
    output_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    boundary_discontinuity: float


def boundary_discontinuity(audio: object, chunk_samples: int, window: int = 8) -> float:
    torch = _torch()
    if audio.shape[-1] <= chunk_samples:
        return 0.0
    vals = []
    for pos in range(chunk_samples, audio.shape[-1], chunk_samples):
        left = audio[..., max(0, pos - window) : pos]
        right = audio[..., pos : min(audio.shape[-1], pos + window)]
        if left.numel() and right.numel():
            vals.append((right.mean() - left.mean()).abs())
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean().item())


def compare_outputs(reference: object, candidate: object, chunk_samples: int) -> ParityMetrics:
    torch = _torch()
    n = min(reference.shape[-1], candidate.shape[-1])
    ref = reference[..., :n]
    cand = candidate[..., :n]
    diff = (ref - cand).abs()
    return ParityMetrics(
        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
        output_shape=tuple(candidate.shape),
        reference_shape=tuple(reference.shape),
        boundary_discontinuity=boundary_discontinuity(candidate, chunk_samples),
    )


def run_tensorized_stream(step: object, wav: object, speaker_embed: object, state: tuple[object, ...], chunk_samples: int, include_warmup: bool = True) -> tuple[object, tuple[object, ...]]:
    torch = _torch()
    ys = []
    cur_state = state
    total = wav.shape[-1]
    padded_total = ((total + chunk_samples - 1) // chunk_samples) * chunk_samples
    if padded_total > total:
        wav = torch.nn.functional.pad(wav, (0, padded_total - total))
    with torch.no_grad():
        for offset in range(0, padded_total, chunk_samples):
            out = step(wav[..., offset : offset + chunk_samples], speaker_embed, *cur_state)
            y, *cur_state = out
            ready = True
            if len(cur_state) > 1 and hasattr(cur_state[1], "numel") and cur_state[1].numel() == 1:
                ready = bool(cur_state[1].item())
            if include_warmup or ready:
                ys.append(y)
    if not ys:
        return wav.new_zeros(wav.shape[:-1] + (0,)), tuple(cur_state)
    y_stream = torch.cat(ys, dim=-1)
    if include_warmup:
        y_stream = y_stream[..., :total]
    return y_stream, tuple(cur_state)


def run_legacy_streaming_reference(model: object, wav: object, speaker_embed: object, chunk_samples: int) -> object:
    """Notebook-style reference path.

    This intentionally uses the existing mutable streaming implementation and is
    kept outside export boundaries.
    """
    torch = _torch()
    encoder = getattr(model.encoder, "net", model.encoder)
    frame_samples = 320
    chunk_frames = chunk_samples // frame_samples
    lookahead_size = 4
    original_length = wav.shape[-1]
    lookahead_samples = lookahead_size * frame_samples
    required = ((original_length + chunk_samples - 1) // chunk_samples) * chunk_samples + lookahead_samples
    audio_padded = torch.nn.functional.pad(wav, (0, max(0, required - original_length)))
    all_audios = []
    embed_stack = torch.empty((wav.shape[0], 512, 0), device=wav.device, dtype=wav.dtype)
    embed_pointer = 0
    with torch.no_grad():
        with encoder.streaming(batch_size=wav.shape[0]):
            with model.decoder.streaming(batch_size=wav.shape[0]):
                for offset in range(0, original_length, chunk_samples):
                    frame = audio_padded[:, :, offset : offset + chunk_samples]
                    enc_state = encoder._streaming_state
                    with encoder._context_for_encoder_decoder:
                        emb = encoder.encoder(frame)
                    embed_stack = torch.cat((embed_stack, emb), dim=-1)
                    while embed_stack.shape[-1] - embed_pointer >= lookahead_size + chunk_frames:
                        emb = embed_stack[:, :, embed_pointer : embed_pointer + lookahead_size + chunk_frames]
                        embed_pointer += chunk_frames
                        assert enc_state.graphed_tr_enc is not None
                        (emb,) = enc_state.graphed_tr_enc(emb, lookahead_size)
                        z_embed, *_ = encoder.quantizer(emb)
                        if isinstance(z_embed, tuple):
                            z_embed = z_embed[0]
                        z_embed = z_embed[:, :, :chunk_frames]
                        audio_chunk, _ = model.decoder(z_embed, speaker_embed)
                        all_audios.append(audio_chunk)
    if not all_audios:
        return wav.new_zeros(wav.shape)
    return torch.cat(all_audios, dim=-1)[..., :original_length]
