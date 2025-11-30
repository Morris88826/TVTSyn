import os
import sys
sys.path.append('..')
import torch
import random
import librosa
import argparse
import pandas as pd
from scipy.io import wavfile
from IPython import display as disp
from pathlib import Path
from tqdm import tqdm
from src.modules import ResynthesisWithQuantAndTVTimbreModule, SpeakerModule

MAX_WAV_VALUE = 32768.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = {
    # source datasets
    'artic': "/data/mtseng/L2Artic/all_data_for_ac_vc",
    'l2artic': "/data/mtseng/L2Artic/all_data_for_ac_vc",
    'vctk': "/data/mtseng/VCTK-Corpus/wav48",
    
    # target datasets
    'emime-eng': "/data/mtseng/EMIME/UEDIN_bilingual_96kHz_data/English_talkers",
    'emime-fin': "/data/mtseng/EMIME/UEDIN_bilingual_96kHz_data/Finnish_talkers",
    'emime-ger': "/data/mtseng/EMIME/UEDIN_bilingual_96kHz_data/German_talkers",
    'emime-man': "/data/mtseng/EMIME/UEDIN_bilingual_96kHz_data/Mandarin_talkers",
} 


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

def inference(resynthesizer, source_audio_path, tgt_embed, model_name, use_projector=False, sample_rate=16000, right_context=4, device='cuda'):
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

def inference_latransformer(resynthesizer, source_audio_path, tgt_embed, sample_rate=16000, device='cuda'):
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


def get_synthesizer_model(model_name, device='cuda'):
    if model_name == "DarkStream-quant-TVT":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre.ckpt",
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
    elif model_name == "DarkStream-quant-TVT-q256":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_q256.ckpt",
            encoder_path=None,
            map_location=device
        )
    elif model_name == "DarkStream-quant-TVT-q1024":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "../ckpts/resynthesis_quant_and_tvtimbre_q1024.ckpt",
            encoder_path=None,
            map_location=device
        )
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
    
    # resynthesizer = resynthesizer.to(device)
    return resynthesizer


def run_rank(rank: int, world_size: int, device: str, model_name: str, fpaths: list, speaker_embeddings: pd.DataFrame, out_dir: str):
    # speaker_encoder = SpeakerModule(device=device)
    # use_projector = False
    
    resynthesizer = get_synthesizer_model(model_name, device=device)
    resynthesizer = resynthesizer.eval()

    sharded_fpaths = fpaths[rank::world_size]
    

    for source_audio_path in tqdm(sharded_fpaths, total=len(sharded_fpaths), desc=f"Rank {rank}"):
        assert os.path.exists(source_audio_path), f"File {source_audio_path} does not exist!"

        out_wav_path = os.path.join(out_dir, "wav", Path(source_audio_path).stem + ".wav")
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        if os.path.exists(out_wav_path):
            continue

        # Select a random embedding from the target speaker
        random_id = random.randint(0, len(speaker_embeddings)-1)
        tgt_emb = speaker_embeddings.iloc[random_id]['speaker_embedding'] #.values[0]
        tgt_emb = torch.from_numpy(tgt_emb).unsqueeze(0).to(device)

        if not args.streaming:
            synth_audio = inference_ns(resynthesizer, source_audio_path, tgt_emb, device=device)
        else:
            synth_audio = inference_latransformer(resynthesizer, source_audio_path, tgt_emb, device=device)
        wavfile.write(out_wav_path, 16000, synth_audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    # parser.add_argument("--metadata", type=str, default="/home/mtseng/anony_benchmark/notebooks/evaluation_data.csv", help="Path to the metadata CSV file")
    parser.add_argument("--vpc_split", type=str, default="IEMOCAP_test", help="Path to the metadata CSV file")
    parser.add_argument("--speaker_embeddings", type=str, default="/home/mtseng/anony_benchmark/notebooks/speaker_embeddings.pkl", help="Path to save/load speaker embeddings")
    parser.add_argument("--right_context", type=int, default=4, help="Right context for streaming model, e.g., 4 for 80ms latency")
    parser.add_argument("--streaming", action="store_true", help="Whether to use streaming model")
    parser.add_argument("--out_dir", type=str, default="/mnt/data1/waris/PSI-TAMU/Voice-Privacy-Challenge-2024/data", help="Directory to save the converted audio files")
    args = parser.parse_args()

    model_name = args.model

    if args.vpc_split=="all":
        vpc_splits = [
            "libri_dev_enrolls",
            "libri_dev_trials_m",
            "libri_dev_trials_f",
            "libri_test_enrolls",
            "libri_test_trials_m",
            "libri_test_trials_f",
            "IEMOCAP_dev",
            "IEMOCAP_test",
            "train-clean-360",
        ]
    else:
        vpc_splits = [args.vpc_split]
        
    for split in vpc_splits:
        print("Processing VPC split:", split)
        # metadata_path = args.metadata
        # metadata_df = pd.read_csv(metadata_path)
        out_dir_name = f"{split}_{model_name}"
        out_dir = "{}/{}".format(args.out_dir, out_dir_name)
        if not args.streaming:
            out_dir += "-ns"
        elif args.right_context > 0:
            out_dir += "-{}ms".format(args.right_context*20)
        print("Output directory:", out_dir)
        os.makedirs(out_dir, exist_ok=True)

        speaker_embeddings = pd.read_pickle(args.speaker_embeddings)
        # selected_metadata_df = metadata_df[metadata_df['target_dataset'] == 'emime-eng']

        metadata_path = f"{args.out_dir}/{split}/wav.scp"
        files = open(metadata_path, "r").readlines()
        
        fpaths = []
        for file in files:
            if split == "train-clean-360":
                file = file.strip().split(" ")[-2]
            else:
                file = file.strip().split(" ")[-1]
            fpath = Path(args.out_dir).parent / file
            fpaths.append(str(fpath))
        print("Total # utterances to convert: {}".format(len(fpaths)))


        # Multi-GPU via torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"


        run_rank(local_rank, world_size, device, model_name, fpaths, speaker_embeddings, out_dir)
    print("Evaluation conversion completed.")


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