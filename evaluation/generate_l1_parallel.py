from evaluation.settings import dataset_roots, resolve_device
import os
import sys
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
from training.modules import ResynthesisWithQuantAndTVTimbreModule, SpeakerModule

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio


MAX_WAV_VALUE = 32768.0
device = torch.device(os.environ.get("TVTSYN_RESEARCH_DEVICE", "cpu"))


def get_wav_duration(path: Path) -> float:
    """
    Return duration in seconds without loading audio into memory.
    """
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


def inference_ns(resynthesizer, source_audio_path, tgt_embed, sample_rate=16000, device='cpu'):
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

def inference(resynthesizer, source_audio_path, tgt_embed, model_name, use_projector=False, sample_rate=16000, right_context=4, device='cpu'):
    src_audio, sr = librosa.load(source_audio_path, sr=sample_rate)
    assert sr == sample_rate
    src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)
    # resynthesizer.eval()
    with torch.no_grad():
        original_length = src_audio.shape[-1]
        chunk_size = 3
        frame_samples = 320
        lookahead_size = 7 if "logits" in model_name else right_context

        # The model advances by 'stride' frames each step.
        stride = chunk_size * frame_samples

        # We always need lookahead frames available for the last step.
        lookahead_samples = lookahead_size * frame_samples

        # Number of stride-sized chunks needed to cover original_length
        # (ceil division)
        num_strides = (original_length + stride - 1) // stride

        # Total length required so that:
        # - audio length is a multiple of stride
        # - plus lookahead tail for the final window
        required_length = num_strides * stride + lookahead_samples

        pad_length = max(0, required_length - original_length)

        if pad_length > 0:
            pad = src_audio.new_zeros((*src_audio.shape[:-1], pad_length))
            audio_padded = torch.cat([src_audio, pad], dim=-1)
        else:
            audio_padded = src_audio


        all_audios = []
        all_z_embeds = []
        with resynthesizer.encoder.net.streaming(batch_size=1):
            with resynthesizer.decoder.streaming(batch_size=1):
                for i, offset in enumerate(range(0, original_length, stride)):  # in steps of frame_size, so chunk size is 20ms
                    frame = audio_padded[:, :, offset:offset + stride]
                    if "logits" not in model_name:
                        frame = audio_padded[:, :, offset:offset + stride + lookahead_samples] # add lookahead for quant model
    
                    # print("frame shape", frame.shape)
                    if use_projector:
                        z_embed, _ = resynthesizer.encoder.net(frame)
                    else:
                        z_embed = resynthesizer.encoder.net.encode_to_latent(frame, tr=lookahead_size)
                    if isinstance(z_embed, tuple):
                        z_embed = z_embed[0]
                    if z_embed.shape[-1] == 0:
                        continue
                    if use_projector:
                        z_embed = resynthesizer.projector(z_embed.transpose(1, 2)).transpose(1, 2)

                    if "logits" not in model_name:
                        z_embed = z_embed[:, :, :chunk_size]
                    resynth_audio_chunk, _ = resynthesizer.decoder(z_embed, tgt_embed)

                    all_audios.append(resynth_audio_chunk)
                    all_z_embeds.append(z_embed)
    all_audios = torch.cat(all_audios, dim=-1)
    all_z_embeds = torch.cat(all_z_embeds, dim=-1)

    resyth_audio = all_audios.squeeze().cpu().numpy()
    resyth_audio = resyth_audio * MAX_WAV_VALUE
    resyth_audio = resyth_audio.astype('int16')
    return resyth_audio

def inference_latransformer(resynthesizer, source_audio_path, tgt_embed, sample_rate=16000, device='cpu'):
    src_audio, sr = librosa.load(source_audio_path, sr=sample_rate)
    assert sr == sample_rate
    src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)
    # resynthesizer.eval()
    with torch.no_grad():
        original_length = src_audio.shape[-1]
        chunk_size = 3
        frame_samples = 320
        lookahead_size = 4

        # The model advances by 'stride' frames each step.
        stride = chunk_size * frame_samples

        # We always need lookahead frames available for the last step.
        lookahead_samples = lookahead_size * frame_samples

        # Number of stride-sized chunks needed to cover original_length
        # (ceil division)
        num_strides = (original_length + stride - 1) // stride

        # Total length required so that:
        # - audio length is a multiple of stride
        # - plus lookahead tail for the final window
        required_length = num_strides * stride + lookahead_samples

        pad_length = max(0, required_length - original_length)

        if pad_length > 0:
            pad = src_audio.new_zeros((*src_audio.shape[:-1], pad_length))
            audio_padded = torch.cat([src_audio, pad], dim=-1)
        else:
            audio_padded = src_audio


        all_audios = []
        all_z_embeds = []
        embed_stack = torch.empty((1, 512, 0)).to(device)
        embed_pointer = 0
        with resynthesizer.encoder.net.streaming(batch_size=1):
            with resynthesizer.decoder.streaming(batch_size=1):
                for offset in range(0, original_length, stride + lookahead_samples):  # in steps of frame_size, so chunk size is 20ms
                    frame = audio_padded[:, :, offset:offset + stride + lookahead_samples] # add lookahead for quant model
    
                    enc_state = resynthesizer.encoder.net._streaming_state
                    with resynthesizer.encoder.net._context_for_encoder_decoder:
                        emb = resynthesizer.encoder.net.encoder(frame)

                    embed_stack = torch.cat((embed_stack, emb), dim=-1)
                    if embed_stack.shape[-1] - embed_pointer < lookahead_size + chunk_size:
                        continue

                    while embed_stack.shape[-1] - embed_pointer >= lookahead_size + chunk_size:
                        emb = embed_stack[:, :, embed_pointer:embed_pointer + lookahead_size + chunk_size]
                        embed_pointer += chunk_size

                        assert enc_state.graphed_tr_enc is not None
                        (emb,) = enc_state.graphed_tr_enc(emb, lookahead_size)
                        z_embed, *_ = resynthesizer.encoder.net.quantizer(emb)
                        if isinstance(z_embed, tuple):
                            z_embed = z_embed[0]
                        if z_embed.shape[-1] == 0:
                            continue

                        z_embed = z_embed[:, :, :chunk_size]
                        resynth_audio_chunk, _ = resynthesizer.decoder(z_embed, tgt_embed)

                        all_audios.append(resynth_audio_chunk)
                        all_z_embeds.append(z_embed)
    all_audios = torch.cat(all_audios, dim=-1)
    all_z_embeds = torch.cat(all_z_embeds, dim=-1)

    resyth_audio = all_audios.squeeze().cpu().numpy()
    resyth_audio = resyth_audio * MAX_WAV_VALUE
    resyth_audio = resyth_audio.astype('int16')
    return resyth_audio


def get_synthesizer_model(model_name, device='cpu'):
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


def align_to_source(source_embeds, target_embeds, device='cpu'):
    source_embeds = source_embeds.squeeze(0).cpu().numpy()
    target_embeds = target_embeds.squeeze(0).cpu().numpy()

    distance, path = fastdtw(source_embeds, target_embeds, dist=cosine)

    target_aligned2source = np.zeros((source_embeds.shape[0], target_embeds.shape[1]))
    for (i, j) in path:
        target_aligned2source[i] = target_embeds[j]

    target_aligned2source = torch.from_numpy(target_aligned2source).unsqueeze(0).to(device)

    return target_aligned2source.transpose(1,2).float()


def run_rank(rank: int, world_size: int, device: str, model_name: str, pairs: list):
    speaker_encoder = SpeakerModule(device=device)
    # use_projector = False
    
    resynthesizer = get_synthesizer_model(model_name, device=device)
    resynthesizer = resynthesizer.eval()

    denoiser_model = pretrained.dns64().to(device)

    sharded_df = pairs[rank::world_size]
    

    for (nns_file, ns_file) in tqdm(sharded_df, total=len(sharded_df)):
        source_audio_path = str(nns_file)
        target_audio_path = str(ns_file)

        assert os.path.exists(source_audio_path), f"File {source_audio_path} does not exist!"
        assert os.path.exists(target_audio_path), f"File {target_audio_path} does not exist!"


        src_audio, _ = librosa.load(source_audio_path, sr=16000)
        src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)

        tgt_audio, _ = librosa.load(target_audio_path, sr=16000)
        tgt_audio = torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            _, z_embed_nns, _ = resynthesizer.encoder(src_audio)
            _, z_embed_ns, _ = resynthesizer.encoder(tgt_audio)

        ## Align the lengths of z_embed_nns and z_embed_ns
        z_embed = align_to_source(z_embed_nns.transpose(1,2), z_embed_ns.transpose(1,2), device=device)

        with torch.no_grad():
            spkr_embed = speaker_encoder(src_audio.squeeze(1))

            audio = resynthesizer.decoder.decode(z_embed, spkr_embed, pitch_rate=1.0, energy_rate=1.0)
        

        out_wav_path = Path(source_audio_path).parent.parent / "L1_aligned_wav" / Path(source_audio_path).name
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        if os.path.exists(out_wav_path):
            continue

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
    parser.add_argument("--device", default=os.environ.get("TVTSYN_RESEARCH_DEVICE", "cpu"))
    parser.add_argument("--data-root", default=None, help="Corpus root; alternatively set TVTSYN_DATA_ROOT or TVTSYN_DATASETS_JSON")
    args = parser.parse_args()
    data_dir = dataset_roots(args.data_root)

    model_name = args.model
    corpus_name = args.corpus

    pairs = []
    if corpus_name == "l2arctic":
        data_dir = dataset_roots(args.data_root)["l2en"]
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
        data_dir_indictts = dataset_roots(args.data_root)["indictts"]
        indictts_dir = Path(data_dir_indictts)
        if indictts_dir.exists():
            indic_tts_files = sorted(list(indictts_dir.rglob("**/english/wav/*.wav")))
            for nns_file in indic_tts_files:
                ns_file_path = nns_file.parent.parent.parent / "native_f5tts/wav" / nns_file.name
                if ns_file_path.exists():
                    pairs.append((nns_file, ns_file_path))

            print(f"Found {len(pairs)} non-native to native speaker pairs in IndicTTS dataset")
    elif corpus_name == "nptel":
        data_dir_nptel = dataset_roots(args.data_root)["nptel"]
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
    device = resolve_device(args.device, local_rank, world_size)

    run_rank(local_rank, world_size, device, model_name, pairs)


"""
torchrun --nproc_per_node=4 evalution.py --model DarkStream-quant-TVT \
      --streaming --right_context 4 --metadata evaluation/evaluation_data.csv \
        --speaker_embeddings evaluation/speaker_embeddings.pkl \
            --out_dir outputs/evaluation

in case port is occupied:

torchrun --standalone --nproc_per_node=4 --master-port=29666 evaluation.py \
    --model DarkStream-quant-TVT-nogtm --metadata evaluation_data.csv \
        --speaker_embeddings speaker_embeddings.pkl --out_dir=ICLR26Results
"""
