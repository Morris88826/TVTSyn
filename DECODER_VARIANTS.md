# Waveform decoder variants

`ResynthesisWithQuantAndTVTimbreModule` accepts `decoder_type` with these
values:

- `seanet` (default, preserves old configs and checkpoints)
- `frame_linear`
- `subframe_linear`
- `hybrid_linear_cnn`
- `istft`

The new decoders consume the conditioned frame representation produced by
`FrameSynthesizerWOExciter` in `[B, C, T]` layout. Timbre, pitch, and energy
conditioning therefore follow the existing path unchanged.

Example Hydra configuration:

```yaml
model:
  init_args:
    decoder_type: frame_linear
    waveform_decoder_args:
      input_dimension: 512
      frame_hop_samples: 320
      decoder_dim: 512
      decoder_layers: 4
      decoder_heads: 8
      decoder_context_frames: 100
      decoder_right_context_frames: 0
      use_post_filter: true
```

The frame, subframe, and hybrid decoders emit exactly `T * 320` samples for
`T` input frames. Enter streaming mode with the existing
`decoder.streaming(batch_size)` API; transformer KV caches and causal
convolution history persist across calls. `ISTFTDecoder` is experimental and
full-utterance only. It raises `NotImplementedError` in streaming mode until a
stateful overlap-add implementation is added.

Run `benchmark_decoder.py --decoder_type frame_linear --chunk_frames 3` to
measure decoder-only first-chunk latency, average latency, RTF, parameter count,
and output length.
