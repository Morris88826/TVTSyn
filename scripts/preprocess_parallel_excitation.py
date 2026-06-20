#!/usr/bin/env python3
"""Precompute target WORLD excitation tracks for parallel wav2wav training."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback
import time

import librosa
import numpy as np
import pyworld
import soundfile as sf


def output_path(target: Path, directory_name: str = "excitation") -> Path:
    return target.parent.parent / directory_name / f"{target.stem}.npz"


def frame_rms(waveform: np.ndarray, hop_length: int) -> np.ndarray:
    frames = max(1, int(np.ceil(waveform.shape[0] / hop_length)))
    padded = np.pad(waveform, (0, frames * hop_length - waveform.shape[0]))
    return np.sqrt(
        np.mean(padded.reshape(frames, hop_length) ** 2, axis=-1) + 1e-12
    )


def analyze_target(
    source_wav: str,
    target_wav: str,
    sample_rate: int,
    hop_length: int,
    frame_period_ms: float,
    f0_method: str,
    f0_floor: float,
    f0_ceil: float,
    save_coded_ap: bool,
) -> dict[str, np.ndarray | int | float | str]:
    waveform, native_rate = sf.read(target_wav, always_2d=True, dtype="float64")
    waveform = waveform.mean(axis=1)
    if native_rate != sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=native_rate, target_sr=sample_rate
        ).astype(np.float64)
    if f0_method == "harvest":
        f0, time_axis = pyworld.harvest(
            waveform,
            sample_rate,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            frame_period=frame_period_ms,
        )
    else:
        f0, time_axis = pyworld.dio(
            waveform,
            sample_rate,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            frame_period=frame_period_ms,
        )
    f0 = pyworld.stonemask(waveform, f0, time_axis, sample_rate)
    energy = frame_rms(waveform, hop_length)
    arrays: dict[str, np.ndarray] = {
        "f0_hz": f0,
        "vuv": (f0 > 0).astype(np.float64),
        "log_f0": np.where(f0 > 0, np.log(np.maximum(f0, 1e-7)), 0.0),
        "energy": energy,
        "log_energy": np.log(np.maximum(energy, 1e-7)),
    }
    if save_coded_ap:
        spectral_envelope = pyworld.cheaptrick(
            waveform, f0, time_axis, sample_rate
        )
        aperiodicity = pyworld.d4c(
            waveform, f0, time_axis, sample_rate
        )
        arrays["coded_ap"] = pyworld.code_aperiodicity(
            aperiodicity, sample_rate
        )
    frame_count = min(array.shape[0] for array in arrays.values())
    result: dict[str, np.ndarray | int | float | str] = {
        name: np.asarray(array[:frame_count], dtype=np.float32)
        for name, array in arrays.items()
    }
    result.update(
        sample_rate=sample_rate,
        frame_period_ms=frame_period_ms,
        hop_length=hop_length,
        source_wav=source_wav,
        target_wav=target_wav,
        num_samples=waveform.shape[0],
        num_frames=frame_count,
    )
    return result


def process_pair(pair: tuple[str, str], options: dict) -> tuple[str, str]:
    source, target = pair
    destination = output_path(Path(target), options["excitation_dir_name"])
    if destination.exists() and not options["overwrite"]:
        return "skipped", str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = analyze_target(
        source,
        target,
        options["sample_rate"],
        options["hop_length"],
        options["frame_period_ms"],
        options["f0_method"],
        options["f0_floor"],
        options["f0_ceil"],
        options["save_coded_ap"],
    )
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result)
    temporary.replace(destination)
    return "processed", str(destination)


def read_manifest(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"{path}:{line_number}: expected source.wav|target.wav")
        pairs.append((parts[0], parts[1]))
    return pairs


def validate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        print(f"f0_hz shape: {data['f0_hz'].shape}")
        print(f"vuv shape: {data['vuv'].shape}")
        print(f"log_energy shape: {data['log_energy'].shape}")
        if "coded_ap" in data:
            print(f"coded_ap shape: {data['coded_ap'].shape}")
        voiced = data["f0_hz"][data["vuv"] > 0.5]
        print(f"voiced percentage: {100.0 * data['vuv'].mean():.2f}%")
        if voiced.size:
            print(
                "voiced f0 min/median/max: "
                f"{voiced.min():.2f}/{np.median(voiced):.2f}/{voiced.max():.2f}"
            )
        print(
            f"energy range: {data['energy'].min():.6f} .. "
            f"{data['energy'].max():.6f}"
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--frame_period_ms", type=float, default=20.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_coded_ap", action="store_true")
    parser.add_argument("--f0_method", choices=("harvest", "dio"), default="harvest")
    parser.add_argument("--f0_floor", type=float, default=50.0)
    parser.add_argument("--f0_ceil", type=float, default=800.0)
    parser.add_argument("--excitation_dir_name", default="excitation")
    parser.add_argument("--validate_npz", type=Path)
    args = parser.parse_args()
    if args.validate_npz is not None:
        validate_npz(args.validate_npz)
        return
    if args.metadata_path is None:
        parser.error("--metadata_path is required unless --validate_npz is used")

    pairs = read_manifest(args.metadata_path)
    options = vars(args).copy()
    options.pop("metadata_path")
    options.pop("validate_npz")
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    started = time.monotonic()
    print(
        f"starting excitation preprocessing: files={len(pairs)} "
        f"workers={max(1, args.num_workers)} method={args.f0_method}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = {
            executor.submit(process_pair, pair, options): pair for pair in pairs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                status, _ = future.result()
                counts[status] += 1
            except Exception:
                counts["failed"] += 1
                source, target = futures[future]
                failures.append(f"{source}|{target}\n{traceback.format_exc()}")
            if completed % max(1, args.progress_every) == 0 or completed == len(pairs):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = completed / elapsed
                remaining = (len(pairs) - completed) / max(rate, 1e-6)
                print(
                    f"progress: {completed}/{len(pairs)} "
                    f"({100.0 * completed / max(len(pairs), 1):.2f}%) "
                    f"processed={counts['processed']} "
                    f"skipped={counts['skipped']} failed={counts['failed']} "
                    f"rate={rate:.2f} files/s eta={remaining / 3600:.2f}h",
                    flush=True,
                )
    failure_path = args.metadata_path.with_suffix(".excitation_failures.log")
    failure_path.write_text("\n".join(failures))
    print(f"total files: {len(pairs)}")
    print(f"processed: {counts['processed']}")
    print(f"skipped: {counts['skipped']}")
    print(f"failed: {counts['failed']}")
    print(f"failure log path: {failure_path}")


if __name__ == "__main__":
    main()
