"""Decoder-only latency benchmark for 50 Hz waveform synthesis."""

import argparse
import time

import torch

from tvtsyn.components.data_classes import DecSeanetNCArgs, WaveformDecoderArgs
from tvtsyn.components.waveform_decoders import create_waveform_decoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder_type",
        choices=[
            "seanet",
            "frame_linear",
            "subframe_linear",
            "hybrid_linear_cnn",
            "istft",
        ],
        default="frame_linear",
    )
    parser.add_argument("--chunk_frames", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    decoder = create_waveform_decoder(
        args.decoder_type,
        seanet_args=DecSeanetNCArgs(causal=True),
        decoder_args=WaveformDecoderArgs(),
    ).to(device=device, dtype=dtype)
    decoder.eval()
    x = torch.randn(1, 512, args.chunk_frames, device=device, dtype=dtype)

    streaming_context = (
        torch.no_grad()
        if args.decoder_type == "istft"
        else decoder.streaming(1)
    )
    all_timings = []
    with torch.inference_mode(), streaming_context:
        for _ in range(args.warmup + args.iterations):
            synchronize(device)
            start = time.perf_counter()
            wav = decoder(x)
            synchronize(device)
            all_timings.append(time.perf_counter() - start)

    timings = all_timings[args.warmup:]
    chunk_seconds = args.chunk_frames / 50.0
    first_ms = all_timings[0] * 1000
    average_ms = sum(timings) / len(timings) * 1000
    params = sum(parameter.numel() for parameter in decoder.parameters())
    print(f"decoder_type: {args.decoder_type}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"parameters: {params:,}")
    print(f"chunk_frames: {args.chunk_frames} ({chunk_seconds * 1000:.1f} ms)")
    print(f"output_samples: {wav.shape[-1]}")
    print(f"first_chunk_ms: {first_ms:.3f}")
    print(f"average_chunk_ms: {average_ms:.3f}")
    print(f"rtf: {(average_ms / 1000) / chunk_seconds:.4f}")


if __name__ == "__main__":
    main()
