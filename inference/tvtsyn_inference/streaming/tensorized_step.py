from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .tensorized_modules import functional_module, functional_streaming_add
from .tensorized_transformer import functional_projected_transformer, functional_transformer


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for TVTSynStreamingStep. Install project requirements "
            "or run in the TVTSyn training environment."
        ) from exc
    return torch


class TVTSynStreamingStep(_torch().nn.Module):  # type: ignore[misc]
    """Export-facing fixed-shape streaming step.

    The first four state tensors implement the encoder lookahead frontier:
    calls seen, current ready flag, rolling encoder-latent history, and the
    total number of encoded latent frames observed so far. The model emits a
    fixed-size audio chunk every call; calls before the first lookahead window
    is complete emit zeros.
    """

    def __init__(self, model: Any, chunk_samples: int = 1600, sample_rate: int = 16000):
        torch = _torch()
        super().__init__()
        if chunk_samples not in (320, 640, 960, 1280, 1600, 2240):
            raise ValueError(f"Supported chunk sizes are 320, 640, 960, 1280, 1600, and 2240 samples; got {chunk_samples}.")
        source_model = model.eval()
        self.__dict__["_source_model"] = source_model
        if hasattr(source_model, "encoder") and hasattr(source_model, "decoder"):
            self.encoder_net = source_model.encoder.net if hasattr(source_model.encoder, "net") else source_model.encoder
            self.decoder = source_model.decoder
        else:
            self.fallback_model = source_model
        self.chunk_samples = int(chunk_samples)
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(sample_rate // 50)
        self.chunk_frames = int(chunk_samples // self.frame_samples)
        self.lookahead_frames = self._infer_lookahead_frames()
        # Number of latent frames retained so that the oldest exportable
        # [chunk + right-context] window is available at a fixed slice.
        self.history_frames = int((2 * self.chunk_frames) + self.lookahead_frames)
        if self.lookahead_frames > 0:
            self.history_window_start = int(((self.lookahead_frames - 1) % self.chunk_frames) + 1)
        else:
            self.history_window_start = int(self.chunk_frames)
        self.register_buffer("_zero", torch.zeros(1), persistent=False)
        self._state_names: list[str] = []
        self._encoder_transformer_layout: dict[str, Any] | None = None
        self._decoder_transformer_layouts: dict[str, dict[str, Any] | None] = {}
        self._tvt_offset_idx: int | None = None
        self._init_state_layout()

    def _apply(self, fn, recurse=True):
        # Cached layout dictionaries refer to buffers; keep those references valid
        # when Module.to() replaces buffers during a device/dtype transfer.
        old = {id(value): name for name, value in self.named_buffers()}
        result = super()._apply(fn, recurse=recurse)
        current = dict(self.named_buffers())
        def update(value):
            if id(value) in old:
                return current[old[id(value)]]
            if isinstance(value, dict):
                return {k: update(v) for k, v in value.items()}
            if isinstance(value, list):
                return [update(v) for v in value]
            return value
        self._encoder_transformer_layout = update(self._encoder_transformer_layout)
        self._decoder_transformer_layouts = update(self._decoder_transformer_layouts)
        return result

    def _infer_lookahead_frames(self) -> int:
        try:
            attn = self.encoder_net.encoder_transformer.transformer.layers[0].self_attn
            return int(getattr(attn, "right_context", 0) or 0)
        except Exception:
            return 0

    def _init_state_layout(self) -> None:
        if not self._has_tvtsyn_parts():
            return
        try:
            from .state_schema import make_tvtsyn_state_schema

            torch = _torch()
            schema = make_tvtsyn_state_schema(self, 1, self.chunk_samples, torch.device("cpu"), torch.float32)
            self._state_names = schema.state_names
            self._encoder_transformer_layout = self._make_transformer_layout(
                "encoder.transformer", self.encoder_net.encoder_transformer.transformer, self.lookahead_frames
            )
            frame = getattr(self.decoder, "decoder_transformer", None)
            if frame is not None:
                self._decoder_transformer_layouts = {
                    "b1_transformer": self._make_transformer_layout("decoder.b1_transformer", frame.b1_transformer, 0),
                    "b2_transformer": self._make_transformer_layout("decoder.b2_transformer", frame.b2_transformer, 0),
                    "b3_transformer": self._make_transformer_layout("decoder.b3_transformer", frame.b3_transformer, 0),
                }
                name_to_idx = {name: idx for idx, name in enumerate(self._state_names)}
                self._tvt_offset_idx = name_to_idx.get("decoder.decoder_transformer.timbre_encoder.pos_offset")
                timbre = frame.timbre_encoder
                half = int(timbre.head_dim) // 2
                self.register_buffer(
                    "_task_b_tvt_rope_inv_freq",
                    1.0 / (float(timbre.rope_base) ** (torch.arange(half, dtype=torch.float32) / half)),
                    persistent=False,
                )
                self.register_buffer(
                    "_task_b_tvt_time_offsets", torch.arange(self.chunk_frames, dtype=torch.float32),
                    persistent=False,
                )
        except Exception:
            self._state_names = []
            self._encoder_transformer_layout = None
            self._decoder_transformer_layouts = {}
            self._tvt_offset_idx = None

    def _make_transformer_layout(self, prefix: str, transformer: Any, tr: int) -> dict[str, Any] | None:
        torch = _torch()
        name_to_idx = {name: idx for idx, name in enumerate(self._state_names)}
        total_t = self.chunk_frames + int(tr)
        position_name = f"_task_b_{prefix.replace('.', '_')}_position_offsets"
        self.register_buffer(position_name, torch.arange(total_t, dtype=torch.long), persistent=False)
        layers = []
        for layer_idx, _layer in enumerate(getattr(transformer, "layers", []) or []):
            k_name = f"{prefix}.layers.{layer_idx}.self_attn.k_cache"
            v_name = f"{prefix}.layers.{layer_idx}.self_attn.v_cache"
            e_name = f"{prefix}.layers.{layer_idx}.self_attn.end_offset"
            l_name = f"{prefix}.layers.{layer_idx}.offset"
            if k_name not in name_to_idx or v_name not in name_to_idx or e_name not in name_to_idx:
                return None
            attn = _layer.self_attn
            capacity = int(getattr(attn, "context", 1) or 1)
            head_dim = int(attn.embed_dim) // int(attn.num_heads)
            stem = f"_task_b_{prefix.replace('.', '_')}_{layer_idx}"
            buffers = {
                "write_offsets": torch.arange(total_t - int(tr), dtype=torch.long),
                "cache_positions": torch.arange(capacity, dtype=torch.long),
                "future_offsets": torch.arange(total_t - int(tr), total_t, dtype=torch.long),
                "query_offsets": torch.arange(total_t, dtype=torch.long),
                "query_offsets_float": torch.arange(total_t, dtype=torch.float32),
                "rope_inv_freq": torch.exp(
                    torch.arange(head_dim // 2, dtype=torch.float32)
                    * (-__import__("math").log(float(getattr(getattr(attn, "rope", None), "max_period", 10000.0))) * 2 / head_dim)
                ),
            }
            constants = {}
            for key, value in buffers.items():
                buffer_name = f"{stem}_{key}"
                self.register_buffer(buffer_name, value, persistent=False)
                constants[key] = getattr(self, buffer_name)
            layers.append({
                "k_idx": name_to_idx[k_name], "v_idx": name_to_idx[v_name],
                "end_idx": name_to_idx[e_name], "layer_offset_idx": name_to_idx.get(l_name),
                "constants": constants,
            })
        return {
            "layers": layers, "offset_idx": name_to_idx.get(f"{prefix}.offset"),
            "position_offsets": getattr(self, position_name),
        }

    @property
    def output_samples(self) -> int:
        return self.chunk_samples

    def _real_model(self) -> Any:
        return self.__dict__.get("_source_model", self)

    def _has_tvtsyn_parts(self) -> bool:
        return hasattr(self, "encoder_net") and hasattr(self, "decoder")

    def _normalize_encoder_frames(self, emb: Any) -> Any:
        torch = _torch()
        if emb.shape[-1] < self.chunk_frames:
            emb = torch.nn.functional.pad(emb, (0, self.chunk_frames - emb.shape[-1]))
        return emb[..., : self.chunk_frames]

    def _encode_cnn_chunk(self, wav_chunk: Any, state_tensors: tuple[Any, ...], next_state: list[Any]) -> Any:
        # This mirrors the notebook's first phase: run only the waveform encoder
        # front-end, then defer transformer lookahead until enough latent frames
        # are present in the explicit tensor history.
        context = getattr(self.encoder_net, "_context_for_encoder_decoder", nullcontext())
        with context:
            emb = functional_module(self.encoder_net.encoder, wav_chunk, "encoder_net.encoder", self._state_names, state_tensors, next_state)
        return self._normalize_encoder_frames(emb)

    def _run_encoder_transformer_and_quantizer(self, emb_window: Any, state_tensors: tuple[Any, ...], next_state: list[Any]) -> Any:
        if self._encoder_transformer_layout is not None:
            out = functional_projected_transformer(
                self.encoder_net.encoder_transformer, emb_window, self.lookahead_frames, state_tensors, next_state, self._encoder_transformer_layout
            )
        else:
            out = self.encoder_net.encoder_transformer(emb_window, self.lookahead_frames)
        if isinstance(out, (tuple, list)):
            emb = out[0]
        else:
            emb = out
        z_embed, *_ = self.encoder_net.quantizer(emb)
        if isinstance(z_embed, tuple):
            z_embed = z_embed[0]
        return z_embed[..., : self.chunk_frames]

    def _rope_cache_with_offset(self, start: Any, length: int, dim: int, device: Any, base: float) -> tuple[Any, Any]:
        torch = _torch()
        half = dim // 2
        if hasattr(self, "_task_b_tvt_rope_inv_freq") and length == self.chunk_frames:
            inv_freq = self._task_b_tvt_rope_inv_freq
            time_offsets = self._task_b_tvt_time_offsets
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
            time_offsets = torch.arange(length, device=device).float()
        t = time_offsets + start.to(dtype=torch.float32).view(1)
        freqs = torch.einsum("l,d->ld", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def precompute_speaker(self, speaker_embed: Any) -> tuple[Any, Any, Any]:
        torch = _torch()
        module = self.decoder.decoder_transformer.timbre_encoder
        if getattr(module, "no_gtm", False):
            raise RuntimeError("Speaker precompute currently requires the GTM cross-attention path.")
        K_seq, V_seq = module.gtm(speaker_embed)
        B = speaker_embed.shape[0]
        K_heads = module.cross._shape(module.cross.k_proj(K_seq), B)
        V_heads = module.cross._shape(module.cross.v_proj(V_seq), B)
        global_proj = module.speaker_proj(torch.nn.functional.normalize(speaker_embed, dim=-1))
        return K_heads, V_heads, global_proj

    def _run_tvt_timbre_precomputed(
        self,
        module: Any,
        x: Any,
        speaker_k: Any,
        speaker_v: Any,
        global_proj: Any,
        state_tensors: tuple[Any, ...],
        next_state: list[Any],
    ) -> Any:
        torch = _torch()
        from tvtsyn.components.tv_timbre import linear_interp, rope_apply, slerp

        B, _, T = x.shape
        q_new = module.q_proj(x).transpose(1, 2)
        if getattr(module, "no_gate", False):
            alpha = torch.full((B, T, 1), float(module.fixed_alpha), device=x.device, dtype=x.dtype)
        else:
            alpha = module.gate(x).transpose(1, 2)
        offset = state_tensors[self._tvt_offset_idx] if self._tvt_offset_idx is not None else torch.zeros(1, device=x.device, dtype=torch.long)
        q = module.cross._shape(module.cross.q_proj(q_new), B)
        rope_q = self._rope_cache_with_offset(offset, T, module.head_dim, x.device, float(module.rope_base))
        q = rope_apply(q, rope_q[0], rope_q[1])
        scores = torch.matmul(q, speaker_k.transpose(-2, -1)) * float(module.cross.scale)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        attn = torch.matmul(weights, speaker_v).transpose(1, 2).reshape(B, T, -1)
        attn = module.cross.out_dropout(module.cross.out(attn))
        H192 = torch.nn.functional.normalize(module.post(attn.transpose(1, 2)), dim=1)
        g = global_proj.unsqueeze(-1).expand(B, module.d_out, T)
        if getattr(module, "no_slerp", False):
            tv_embed = linear_interp(H192.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)
        else:
            tv_embed = slerp(H192.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)
        if self._tvt_offset_idx is not None:
            next_state[self._tvt_offset_idx] = offset + T
        return tv_embed

    def _run_tvt_timbre(self, module: Any, x: Any, speaker_embed: Any, state_tensors: tuple[Any, ...], next_state: list[Any]) -> Any:
        speaker_k, speaker_v, global_proj = self.precompute_speaker(speaker_embed)
        return self._run_tvt_timbre_precomputed(
            module, x, speaker_k, speaker_v, global_proj, state_tensors, next_state
        )

    def _run_decoder_transformer(
        self, z_embed: Any, speaker_embed: Any, state_tensors: tuple[Any, ...], next_state: list[Any],
        speaker_precomputed: tuple[Any, Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        frame = self.decoder.decoder_transformer
        x = z_embed
        B, _, T = x.shape
        if getattr(frame, "use_tv_timbre", False):
            if speaker_precomputed is None:
                timbre = self._run_tvt_timbre(frame.timbre_encoder, x, speaker_embed, state_tensors, next_state)
            else:
                timbre = self._run_tvt_timbre_precomputed(
                    frame.timbre_encoder, x, *speaker_precomputed, state_tensors, next_state
                )
        else:
            torch = _torch()
            timbre = frame.timbre_encoder(torch.nn.functional.normalize(speaker_embed, dim=-1).unsqueeze(-1)).expand(B, frame.timbre_cond_dim, T)

        if getattr(frame, "conv_layout", False):
            x = x.transpose(1, 2)
        if getattr(frame, "input_proj", None) is not None:
            x = frame.input_proj(x)

        z1 = functional_transformer(frame.b1_transformer, x, 0, state_tensors, next_state, self._decoder_transformer_layouts["b1_transformer"])
        z1 = frame.b1_ln(z1.transpose(1, 2), timbre)

        predicted_pitch = functional_module(
            frame.pitch_energy_predictor.pitchPred.model,
            z1,
            "decoder.decoder_transformer.pitch_energy_predictor.pitchPred.model",
            self._state_names,
            state_tensors,
            next_state,
        )
        predicted_energy = functional_module(
            frame.pitch_energy_predictor.energyPred.model,
            z1,
            "decoder.decoder_transformer.pitch_energy_predictor.energyPred.model",
            self._state_names,
            state_tensors,
            next_state,
        )
        avg_pitch, avg_energy = None, None
        if getattr(frame, "use_pitch_energy_predictor", False):
            pitch = functional_module(frame.pitchEmbed, predicted_pitch, "decoder.decoder_transformer.pitchEmbed", self._state_names, state_tensors, next_state)
            energy = functional_module(frame.energyEmbed, predicted_energy, "decoder.decoder_transformer.energyEmbed", self._state_names, state_tensors, next_state)
            z1 = functional_streaming_add(z1, pitch, "decoder.decoder_transformer.pitch_add", self._state_names, state_tensors, next_state)
            z1 = functional_streaming_add(z1, energy, "decoder.decoder_transformer.energy_add", self._state_names, state_tensors, next_state)

        z2 = functional_transformer(frame.b2_transformer, z1.transpose(1, 2), 0, state_tensors, next_state, self._decoder_transformer_layouts["b2_transformer"])
        z = frame.b2_ln(z2.transpose(1, 2), timbre)
        z3 = functional_transformer(frame.b3_transformer, z.transpose(1, 2), 0, state_tensors, next_state, self._decoder_transformer_layouts["b3_transformer"])
        z = frame.b3_ln(z3.transpose(1, 2), timbre)
        y = frame.output_proj(z.transpose(1, 2)).transpose(1, 2)
        return y, (avg_pitch, predicted_pitch, avg_energy, predicted_energy)

    def _decode_latent_chunk(
        self, z_embed: Any, speaker_embed: Any, state_tensors: tuple[Any, ...], next_state: list[Any],
        speaker_precomputed: tuple[Any, Any, Any] | None = None,
    ) -> Any:
        torch = _torch()
        try:
            from tvtsyn.utils.compile import no_compile
        except Exception:
            no_compile = nullcontext
        with no_compile():
            if getattr(self.decoder, "decoder_transformer", None) is not None and self._decoder_transformer_layouts:
                emb, _ = self._run_decoder_transformer(
                    z_embed, speaker_embed, state_tensors, next_state, speaker_precomputed
                )
            else:
                emb = z_embed
            y_chunk = functional_module(self.decoder.decoder, emb, "decoder.decoder", self._state_names, state_tensors, next_state)
        if y_chunk.shape[-1] < self.chunk_samples:
            y_chunk = torch.nn.functional.pad(y_chunk, (0, self.chunk_samples - y_chunk.shape[-1]))
        return y_chunk[..., : self.chunk_samples]

    def _fallback_chunk(self, wav_chunk: Any, speaker_embed: Any) -> Any:
        torch = _torch()
        y_chunk = self.fallback_model(wav_chunk, speaker_embed)
        if isinstance(y_chunk, tuple):
            y_chunk = y_chunk[0]
        if y_chunk.shape[-1] < self.chunk_samples:
            y_chunk = torch.nn.functional.pad(y_chunk, (0, self.chunk_samples - y_chunk.shape[-1]))
        return y_chunk[..., : self.chunk_samples]

    def _streaming_tvtsyn_chunk(
        self, wav_chunk: Any, speaker_embed: Any, history: Any, state_tensors: tuple[Any, ...], next_state: list[Any],
        speaker_precomputed: tuple[Any, Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        torch = _torch()
        emb = self._encode_cnn_chunk(wav_chunk, state_tensors, next_state)
        next_history = torch.cat([history[..., self.chunk_frames :], emb], dim=-1)
        window_len = self.chunk_frames + self.lookahead_frames
        emb_window = next_history[..., self.history_window_start : self.history_window_start + window_len]
        z_embed = self._run_encoder_transformer_and_quantizer(emb_window, state_tensors, next_state)
        y_chunk = self._decode_latent_chunk(
            z_embed, speaker_embed, state_tensors, next_state, speaker_precomputed
        )
        return y_chunk, next_history

    def forward_with_precomputed_speaker(
        self, wav_chunk: Any, speaker_k: Any, speaker_v: Any, global_proj: Any, *state_tensors: Any
    ) -> tuple[Any, ...]:
        speaker_placeholder = global_proj.new_zeros((global_proj.shape[0], 1))
        return self._forward_impl(
            wav_chunk, speaker_placeholder, state_tensors, (speaker_k, speaker_v, global_proj)
        )

    def forward(self, wav_chunk: Any, speaker_embed: Any, *state_tensors: Any) -> tuple[Any, ...]:
        return self._forward_impl(wav_chunk, speaker_embed, state_tensors, None)

    def _forward_impl(
        self, wav_chunk: Any, speaker_embed: Any, state_tensors: tuple[Any, ...],
        speaker_precomputed: tuple[Any, Any, Any] | None,
    ) -> tuple[Any, ...]:
        torch = _torch()
        if wav_chunk.dim() != 3:
            raise RuntimeError("wav_chunk must have shape [B, 1, chunk_samples].")
        if wav_chunk.shape[-1] != self.chunk_samples:
            raise RuntimeError("wav_chunk has a different sample length than this fixed-shape step.")
        if len(state_tensors) < 4:
            raise RuntimeError("TVTSynStreamingStep requires the state tuple from init_tvtsyn_state(...).")

        calls_seen = state_tensors[0]
        history = state_tensors[2]
        encoded_seen = state_tensors[3]
        next_calls_seen = calls_seen + 1
        next_encoded_seen = encoded_seen + self.chunk_frames
        ready = (next_encoded_seen >= (self.chunk_frames + self.lookahead_frames)).to(dtype=torch.long)

        next_state = list(state_tensors)

        if self._has_tvtsyn_parts():
            y_candidate, next_history = self._streaming_tvtsyn_chunk(
                wav_chunk, speaker_embed, history, state_tensors, next_state, speaker_precomputed
            )
        else:
            y_candidate = self._fallback_chunk(wav_chunk, speaker_embed)
            next_history = history

        y_chunk = torch.where(ready.view(1, 1, 1) > 0, y_candidate, torch.zeros_like(y_candidate))
        for idx in range(4, len(next_state)):
            name = self._state_names[idx] if idx < len(self._state_names) else ""
            # During lookahead warmup the legacy path advances only the encoder
            # waveform front-end caches. Transformer/decoder caches must not see
            # the synthetic incomplete lookahead window.
            if not name.startswith("encoder_net.encoder."):
                next_state[idx] = torch.where(ready > 0, next_state[idx], state_tensors[idx])
        next_state[0] = next_calls_seen
        next_state[1] = ready
        next_state[2] = next_history
        next_state[3] = next_encoded_seen
        return (y_chunk, *tuple(next_state))
