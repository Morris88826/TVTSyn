from __future__ import annotations

import os
import sys
sys.path.append('..')
import torch
import librosa
import argparse
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from scipy.io import wavfile
from IPython import display as disp
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


from dataclasses import dataclass
from typing import Optional, Tuple

import torch.nn as nn


from src.modules import ResynthesisWithQuantAndTVTimbreModule, SpeakerModule

import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio


MAX_WAV_VALUE = 32768.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_softvc_model(device='cuda'):
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(device)
    acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft", trust_repo=True).to(device)
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft", trust_repo=True).to(device)
    
    hubert = hubert.eval()
    acoustic = acoustic.eval()
    hifigan = hifigan.eval()
    return hubert, acoustic, hifigan


def get_wav_duration(path: Path) -> float:
    """
    Return duration in seconds without loading audio into memory.
    """
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


def inference_ns(resynthesizer, source_audio_path, tgt_embed, sample_rate=16000, device='cuda'):
    src_audio, sr = librosa.load(source_audio_path, sr=sample_rate)
    assert sr == sample_rate
    src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)
    # resynthesizer.eval()
    with torch.no_grad():
        resyth_audio = resynthesizer(src_audio, tgt_embed)
        resyth_audio = resyth_audio.squeeze()
        resyth_audio = resyth_audio * MAX_WAV_VALUE
        resyth_audio = resyth_audio.detach().cpu().numpy().astype('int16')
    return resyth_audio


def get_synthesizer_model(model_name, device='cuda'):
    if model_name == "DarkStream-quant-TVT":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_ft.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-nogate":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_nogate.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-nogtm":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_nogtm.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-noprior":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_noprior.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-noslerp":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_noslerp.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-token12":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_token12.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-token24":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_token24.ckpt",
            encoder_path=None,
            map_location=device
        )
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
    
    # resynthesizer = resynthesizer.to(device)
    return resynthesizer


# -----------------------------
# Polyline simplification (RDP)
# -----------------------------

def _perp_dist(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Perpendicular distance from 'point' to line segment a-b in 2D.
    """
    # If a == b, distance is euclidean to a
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker simplification for 2D polyline.
    points: (N,2) float or int
    eps: tolerance
    Returns a subset of points in order.
    """
    if points.shape[0] <= 2:
        return points

    a = points[0].astype(np.float64)
    b = points[-1].astype(np.float64)

    max_d = -1.0
    idx = -1
    for i in range(1, points.shape[0] - 1):
        d = _perp_dist(points[i].astype(np.float64), a, b)
        if d > max_d:
            max_d = d
            idx = i

    if max_d > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([points[0], points[-1]])


def _compress_repeated_tgt_knots(pts: np.ndarray) -> np.ndarray:
    """
    Ensure no consecutive knots have same target index (avoid tgt_delta==0).
    Overwrite duplicates with later src at same tgt.
    """
    compressed = [pts[0].tolist()]
    for i_src, j_tgt in pts[1:]:
        p_src, p_tgt = compressed[-1]
        if int(j_tgt) == int(p_tgt):
            compressed[-1] = [int(i_src), int(j_tgt)]
        else:
            compressed.append([int(i_src), int(j_tgt)])
    return np.array(compressed, dtype=np.int32)


def _enforce_min_tgt_step(pts: np.ndarray, min_tgt_step: int) -> np.ndarray:
    """
    Enforce that successive knots increase by at least min_tgt_step in target.
    This further prevents tiny segments.
    """
    out = [pts[0]]
    last_src, last_tgt = int(pts[0, 0]), int(pts[0, 1])
    for i in range(1, pts.shape[0]):
        src, tgt = int(pts[i, 0]), int(pts[i, 1])
        if tgt - last_tgt >= min_tgt_step:
            out.append([src, tgt])
            last_src, last_tgt = src, tgt
        else:
            # too close in tgt; skip this knot (keep later ones)
            continue
    if out[-1][0] != int(pts[-1, 0]) or out[-1][1] != int(pts[-1, 1]):
        out.append([int(pts[-1, 0]), int(pts[-1, 1])])
    return np.array(out, dtype=np.int32)


# -----------------------------
# Adaptive segment extraction
# -----------------------------

def get_alignment_segments_adaptive(
    bnf_src: np.ndarray,   # (T_src, C)
    bnf_tgt: np.ndarray,   # (T_tgt, C)
    metric: str = "cosine",
    eps: Optional[float] = None,
    min_tgt_step: int = 1,
    max_knots: Optional[int] = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive knot selection using RDP simplification of the DTW path.

    Returns:
      src_delta: (K,) float32
      tgt_delta: (K,) int32 (>=1)
      tgt_offsets: (K,) int32

    Controls:
      eps: RDP tolerance in "frames" of (src,tgt) space.
           If None, choose a length-aware default.
      min_tgt_step: minimum increase in target index per segment (>=1).
      max_knots: optional cap to prevent too many knots; if exceeded, eps is increased.
    """
    assert bnf_src.ndim == 2 and bnf_tgt.ndim == 2
    T_src, C1 = bnf_src.shape
    T_tgt, C2 = bnf_tgt.shape
    assert C1 == C2

    _, wp = librosa.sequence.dtw(X=bnf_src.T, Y=bnf_tgt.T, metric=metric)
    wp = np.flipud(wp).astype(np.int32)

    # Anchor endpoints
    wp[0] = [0, 0]
    wp[-1] = [T_src - 1, T_tgt - 1]

    # Choose eps if not provided
    if eps is None:
        # 0.5% of total frames, clamped
        eps = 0.005 * float(T_src + T_tgt)
        eps = float(np.clip(eps, 2.0, 20.0))

    # RDP simplify; if too many knots, increase eps until within max_knots
    pts = wp.astype(np.float64)
    simplified = rdp(pts, eps).astype(np.int32)

    if max_knots is not None:
        max_knots = int(max_knots)
        cur_eps = eps
        while simplified.shape[0] > max_knots:
            cur_eps *= 1.25
            simplified = rdp(pts, cur_eps).astype(np.int32)

    # print(f"Adaptive alignment: selected {simplified.shape[0]} knots with eps={cur_eps:.2f}")
    # Make sure knots are strictly usable
    simplified[0] = [0, 0]
    simplified[-1] = [T_src - 1, T_tgt - 1]
    simplified = _compress_repeated_tgt_knots(simplified)
    simplified = _enforce_min_tgt_step(simplified, min_tgt_step=max(1, int(min_tgt_step)))
    simplified = _compress_repeated_tgt_knots(simplified)

    # Segment deltas
    src_delta = (simplified[1:, 0] - simplified[:-1, 0]).astype(np.float32)
    tgt_delta = (simplified[1:, 1] - simplified[:-1, 1]).astype(np.int32)

    keep = tgt_delta > 0
    src_delta = src_delta[keep]
    tgt_delta = tgt_delta[keep]

    tgt_offsets = np.concatenate([[0], np.cumsum(tgt_delta[:-1])]).astype(np.int32)
    return src_delta, tgt_delta, tgt_offsets


# -----------------------------
# Everything else stays the same
# -----------------------------

def quantize_out_lens(out_len_real: torch.Tensor, T_out: int) -> torch.Tensor:
    out_len_real = out_len_real.clamp(min=1e-6)
    base = torch.floor(out_len_real).long().clamp(min=1)
    frac = out_len_real - base.float()
    cur = int(base.sum().item())
    delta = int(T_out) - cur
    if delta > 0:
        idx = torch.argsort(frac, descending=True)[:delta]
        base[idx] += 1
    elif delta < 0:
        idx = torch.argsort(frac, descending=False)
        i = 0
        while delta < 0 and i < idx.numel():
            k = int(idx[i].item())
            if base[k] > 1:
                base[k] -= 1
                delta += 1
            i += 1
    mismatch = int(T_out) - int(base.sum().item())
    if mismatch != 0:
        base[-1] = torch.clamp(base[-1] + mismatch, min=1)
    assert int(base.sum().item()) == int(T_out)
    return base


def normalize_and_cap_out_lens(
    src_delta: torch.Tensor,
    tgt_delta: torch.Tensor,
    T_out: int,
    ratio_min: Optional[float] = 0.5,
    ratio_max: Optional[float] = 3.0,
) -> torch.Tensor:
    out_len_real = src_delta.clamp(min=1e-6)
    out_len_real = out_len_real * (float(T_out) / out_len_real.sum().clamp(min=1e-6))

    if ratio_min is not None or ratio_max is not None:
        denom = tgt_delta.float().clamp(min=1.0)
        ratio = out_len_real / denom
        if ratio_min is not None:
            ratio = torch.clamp(ratio, min=float(ratio_min))
        if ratio_max is not None:
            ratio = torch.clamp(ratio, max=float(ratio_max))
        out_len_real = ratio * denom
        out_len_real = out_len_real * (float(T_out) / out_len_real.sum().clamp(min=1e-6))

    return out_len_real


class SegmentLinearWarp(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, tgt_offsets, tgt_delta, out_len):
        B, T_tgt, C = x.shape
        ys = []
        for k in range(tgt_delta.numel()):
            off = int(tgt_offsets[k].item())
            L_in = int(tgt_delta[k].item())
            L_out = int(out_len[k].item())
            start = off
            end = min(off + L_in - 1, T_tgt - 1)

            if end <= start:
                j = torch.full((L_out,), float(start), device=x.device, dtype=torch.float32)
            elif L_out == 1:
                j = torch.tensor([float(start)], device=x.device, dtype=torch.float32)
            else:
                j = torch.linspace(float(start), float(end), steps=L_out, device=x.device, dtype=torch.float32)

            j0 = torch.floor(j).long().clamp(0, T_tgt - 1)
            j1 = (j0 + 1).clamp(0, T_tgt - 1)
            w = (j - j0.float()).view(1, L_out, 1)

            x0 = x[:, j0, :]
            x1 = x[:, j1, :]
            ys.append((1.0 - w) * x0 + w * x1)

        return torch.cat(ys, dim=1)


@dataclass
class WarpConfig:
    metric: str = "cosine"
    eps: Optional[float] = None          # RDP tolerance; None -> auto
    min_tgt_step: int = 1                # try 1–3
    max_knots: int = 200                 # safety cap
    ratio_min: Optional[float] = 0.5     # clamp local stretch
    ratio_max: Optional[float] = 3.0


def align_native_units_to_non_native_length_adaptive(
    units_non_native: torch.Tensor,  # (1, T_src, C)
    units_native: torch.Tensor,      # (1, T_tgt, C)
    cfg: WarpConfig = WarpConfig(),
) -> torch.Tensor:
    device = units_native.device
    T_src = int(units_non_native.size(1))

    bnf_src = units_non_native.squeeze(0).cpu().numpy()
    bnf_tgt = units_native.squeeze(0).cpu().numpy()

    src_delta, tgt_delta, tgt_offsets = get_alignment_segments_adaptive(
        bnf_src=bnf_src,
        bnf_tgt=bnf_tgt,
        metric=cfg.metric,
        eps=cfg.eps,
        min_tgt_step=cfg.min_tgt_step,
        max_knots=cfg.max_knots,
    )

    src_delta_t = torch.from_numpy(src_delta).to(device=device, dtype=torch.float32)
    tgt_delta_t = torch.from_numpy(tgt_delta).to(device=device, dtype=torch.long)
    tgt_offsets_t = torch.from_numpy(tgt_offsets).to(device=device, dtype=torch.long)

    out_len_real = normalize_and_cap_out_lens(
        src_delta=src_delta_t,
        tgt_delta=tgt_delta_t,
        T_out=T_src,
        ratio_min=cfg.ratio_min,
        ratio_max=cfg.ratio_max,
    )
    out_len = quantize_out_lens(out_len_real, T_out=T_src)

    warp = SegmentLinearWarp().to(device)
    aligned = warp(units_native, tgt_offsets_t, tgt_delta_t, out_len)
    
    # print("aligned:", aligned.shape)

    # enforce exact length (pad/crop deterministically)
    if aligned.size(1) < T_src:
        pad = T_src - aligned.size(1)
        aligned = torch.cat([aligned, aligned[:, -1:, :].expand(-1, pad, -1)], dim=1)
    elif aligned.size(1) > T_src:
        aligned = aligned[:, :T_src, :]
    return aligned


def get_aligned_wav_with_softvc(source_audio_path, target_audio_path, hubert, acoustic, hifigan, device='cuda'):
    source, sr = torchaudio.load(source_audio_path)
    source = torchaudio.functional.resample(source, sr, 16000)
    source = source.unsqueeze(0).to(device)
    
    source_l1, sr = torchaudio.load(target_audio_path)
    source_l1 = torchaudio.functional.resample(source_l1, sr, 16000)
    source_l1 = source_l1.unsqueeze(0).to(device)
    
    with torch.inference_mode():
        # Extract speech units
        units = hubert.units(source)
        # Generate target spectrogram
        
        units_l1 = hubert.units(source_l1)
                
        units_aligned = align_native_units_to_non_native_length_adaptive(units, units_l1)        
        
        mel = acoustic.generate(units_aligned).transpose(1, 2)
        # Generate audio waveform
        target = hifigan(mel)
        
    return target.squeeze()


def run_rank(rank: int, world_size: int, device: str, model_name: str, pairs: list):
    speaker_encoder = SpeakerModule(device=device)
    # use_projector = False
    
    resynthesizer = get_synthesizer_model(model_name, device=device)
    resynthesizer = resynthesizer.eval()
    
    hubert, acoustic, hifigan = get_softvc_model(device=device)

    denoiser_model = pretrained.dns64().to(device)

    sharded_df = pairs[rank::world_size]
    

    for (nns_file, ns_file) in tqdm(sharded_df, total=len(sharded_df)):
        source_audio_path = str(nns_file)
        target_audio_path = str(ns_file)

        assert os.path.exists(source_audio_path), f"File {source_audio_path} does not exist!"
        assert os.path.exists(target_audio_path), f"File {target_audio_path} does not exist!"


        # src_audio, _ = librosa.load(source_audio_path, sr=16000)
        # src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)

        # tgt_audio, _ = librosa.load(target_audio_path, sr=16000)
        # tgt_audio = torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     _, z_embed_nns, _ = resynthesizer.encoder(src_audio)
        #     _, z_embed_ns, _ = resynthesizer.encoder(tgt_audio)

        # ## Align the lengths of z_embed_nns and z_embed_ns
        # z_embed = align_native_units_to_non_native_length_adaptive(z_embed_nns.transpose(1,2), z_embed_ns.transpose(1,2)).transpose(1,2)

        # with torch.no_grad():
        #     spkr_embed = speaker_encoder(src_audio.squeeze(1))

        #     audio = resynthesizer.decoder.decode(z_embed, spkr_embed, pitch_rate=1.0, energy_rate=1.0)
        
        
        target_aligned_wav = get_aligned_wav_with_softvc(source_audio_path, target_audio_path, hubert, acoustic, hifigan, device=device)
        
        # preform vc with resynthesizer
        src_audio, _ = librosa.load(source_audio_path, sr=16000)
        src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            spkr_embed = speaker_encoder(src_audio.squeeze(1))
            src_audio = target_aligned_wav.unsqueeze(0).unsqueeze(0).to(device)
            
            audio = resynthesizer(src_audio, spkr_embed)
        

        out_wav_path = Path(source_audio_path).parent.parent.parent / "golden/wav" / Path(source_audio_path).name
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        # if os.path.exists(out_wav_path):
        #     continue

        # Denoise the audio
        wav = convert_audio(audio.squeeze(1), 16000, denoiser_model.sample_rate, denoiser_model.chin)
        with torch.no_grad():
            denoised = denoiser_model(wav[None])[0]

        synth_wav = denoised.data.cpu().numpy().squeeze()
        synth_wav = synth_wav * MAX_WAV_VALUE
        synth_wav = synth_wav.astype('int16')
        
        wavfile.write(out_wav_path, 16000, synth_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--corpus", type=str, required=True, help="corpus name to use")
    args = parser.parse_args()

    model_name = args.model
    corpus_name = args.corpus

    pairs = []
    if corpus_name == "l2arctic":
        data_dir = "/data/waris/data/Multilingual_dataset/data/L2EN"
        pair_with_all_native_speaker = False

        native_speakers = ["BDL", "RMS", "CLB", "SLT"]
        native_speakers_to_files = {}
        for ns in native_speakers:
            ns_files = list(Path(f"{data_dir}/american/{ns}/wav").rglob("*.wav"))
            # Remove fixed test files (anything greateer than arctic_b0489.wav)
            # ns_files = [f for f in ns_files if (f.stem.split("_")[-1][0] == "a" or int(f.stem.split("_")[-1][1:]) < 489)]
            assert len(ns_files) > 0, f"No files found for native speaker {ns} in {data_dir}/american/{ns}/wav"
            native_speakers_to_files[ns] = ns_files

        nns_accents_to_spkr = {
            "arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
            "chinese": ["BWC", "LXC", "NCC", "TXHC"],
            "korean": ["HJK", "HKK", "YDCK", "YKWK"],
            "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
            "indian": ["ASI", "RRBI", "SVBI", "TNI"],
            "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
            "american": ["BDL", "RMS", "CLB", "SLT"], # self reconstruction
        }
        
        for i, ns in enumerate(native_speakers):
            if not pair_with_all_native_speaker and i > 0:
                break
            ns_files = native_speakers_to_files[ns]
            for ns_file in ns_files:
                for accent, spkrs in nns_accents_to_spkr.items():
                    for spkr in spkrs:
                        nns_file_path = f"{data_dir}/{accent}/{spkr}/wav/{ns_file.name}"

                        if Path(nns_file_path).exists():
                            pairs.append((Path(nns_file_path), ns_file))

        print(f"Found {len(pairs)} non-native to native speaker pairs in {data_dir}")
    elif corpus_name == "indictts":
        data_dir_indictts = "/data/waris/data/IndicTTS"
        indictts_dir = Path(data_dir_indictts)
        if indictts_dir.exists():
            indic_tts_files = sorted(list(indictts_dir.rglob("**/english/wav/*.wav")))
            for nns_file in indic_tts_files:
                ns_file_path = nns_file.parent.parent.parent / "native_f5tts/wav" / nns_file.name
                if ns_file_path.exists():
                    pairs.append((nns_file, ns_file_path))

            print(f"Found {len(pairs)} non-native to native speaker pairs in IndicTTS dataset")
    elif corpus_name == "nptel":
        data_dir_nptel = "/data/waris/data/nptel/data"
        nptel_dir = Path(data_dir_nptel)
        if nptel_dir.exists():
            nptel_files = sorted(list(nptel_dir.rglob("indian/**/wav/*.wav")))
            print(f"Total non-native files found: {len(nptel_files)}")
            for nns_file in tqdm(nptel_files, desc="Processing NPTEL files"):
                #only consider files under 30sec
                wav_dur = get_wav_duration(nns_file)
                if wav_dur > 30.0 or wav_dur < 1.2:
                    continue
                ns_file_path = nptel_dir / "american" / nns_file.relative_to(nptel_dir / "indian")
                if ns_file_path.exists():
                    pairs.append((nns_file, ns_file_path))

            print(f"Found {len(pairs)} non-native to native speaker pairs in NPTEL dataset")
    else:
        raise ValueError("Unknown corpus name: {}".format(corpus_name))


    # Multi-GPU via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"

    run_rank(local_rank, world_size, device, model_name, pairs)


"""
torchrun --nproc_per_node=4 evalution.py --model DarkStream-quant-TVT \
      --streaming --right_context 4 --metadata /home/mtseng/anony_benchmark/notebooks/evaluation_data.csv \
        --speaker_embeddings /home/mtseng/anony_benchmark/notebooks/speaker_embeddings.pkl \
            --out_dir /data/mtseng/ICLR2026/results

in case port is occupied:

torchrun --standalone --nproc_per_node=4 --master-port=29666 evaluation.py \
    --model DarkStream-quant-TVT-nogtm --metadata evaluation_data.csv \
        --speaker_embeddings speaker_embeddings.pkl --out_dir=/data/waris/code/TVTSyn/ICLR26Results
"""