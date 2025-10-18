import os
import torch
import librosa
import argparse
import pandas as pd
from scipy.io import wavfile
from IPython import display as disp
from pathlib import Path
from tqdm import tqdm
from src.modules import ResynthesisWithBottleneckAndTVTimbreModule, ResynthesisWithLogitsAndTVTimbreModule, ResynthesisWithQuantAndTVTimbreModule, SpeakerModule, ResynthesisWithBottleneckModule, ResynthesisWithQuantModule

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


def inference_ns(source_audio_path, tgt_embed, sample_rate=16000):
    src_audio, sr = librosa.load(source_audio_path, sr=sample_rate)
    assert sr == sample_rate
    src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)
    resynthesizer.eval()
    with torch.no_grad():
        resyth_audio = resynthesizer(src_audio, tgt_embed)
        resyth_audio = resyth_audio.squeeze()
        resyth_audio = resyth_audio * MAX_WAV_VALUE
        resyth_audio = resyth_audio.detach().cpu().numpy().astype('int16')
    return resyth_audio

def inference(source_audio_path, tgt_embed, model_name, use_projector=False, sample_rate=16000, right_context=4):
    src_audio, sr = librosa.load(source_audio_path, sr=sample_rate)
    assert sr == sample_rate
    src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)
    resynthesizer.eval()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--metadata", type=str, default="/home/mtseng/anony_benchmark/notebooks/evaluation_data.csv", help="Path to the metadata CSV file")
    parser.add_argument("--speaker_embeddings", type=str, default="/home/mtseng/anony_benchmark/notebooks/speaker_embeddings.pkl", help="Path to save/load speaker embeddings")
    parser.add_argument("--right_context", type=int, default=4, help="Right context for streaming model, e.g., 4 for 80ms latency")
    parser.add_argument("--streaming", action="store_true", help="Whether to use streaming model")
    parser.add_argument("--out_dir", type=str, default="/data/mtseng/ICLR2026/results", help="Directory to save the converted audio files")
    args = parser.parse_args()

    speaker_encoder = SpeakerModule(device="cuda")
    use_projector = False
    model_name = args.model
    if model_name == "DarkStream-logits-TVT":
        resynthesizer = ResynthesisWithLogitsAndTVTimbreModule.load_from_checkpoint(
            "./ckpts/resynthesis_with_logits_and_tvtimbre.ckpt",
            encoder_path=None,
            map_location="cuda"
        )
        use_projector = True
    elif model_name == "DarkStream-bottleneck-TVT":
        # Uncomment the following line to use the KM Resynthesis Module instead
        resynthesizer = ResynthesisWithBottleneckAndTVTimbreModule.load_from_checkpoint(
            "./ckpts/resynthesis_bottleneck_and_tvtimbre.ckpt",
            encoder_path=None,
            decoder_pretrained=None,
            kmeans_path=None,
            map_location="cuda"
        )
    elif model_name == "DarkStream-quant-TVT":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "./ckpts/resynthesis_quant_and_tvtimbre.ckpt",
            encoder_path=None,
            map_location="cuda"
        )
    elif model_name == "DarkStream-bottleneck":
        resynthesizer = ResynthesisWithBottleneckModule.load_from_checkpoint(
            "./ckpts/resynthesis_with_bottleneck.ckpt",
            encoder_path=None,
            decoder_pretrained=None,
            kmeans_path=None,
            map_location="cuda"
        )
    elif model_name == "DarkStream-quant":
        resynthesizer = ResynthesisWithQuantModule.load_from_checkpoint(
            "./ckpts/resynthesis_with_quant.ckpt",
            encoder_path=None,
            map_location="cuda"
        )

    resynthesizer = resynthesizer.eval()

    metadata_path = args.metadata
    metadata_df = pd.read_csv(metadata_path)
    out_dir = "{}/{}".format(args.out_dir, model_name)
    if not args.streaming:
        out_dir += "-ns"
    elif args.right_context > 0:
        out_dir += "-{}ms".format(args.right_context*20)
    print("Output directory:", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    speaker_embeddings = pd.read_pickle(args.speaker_embeddings)
    selected_metadata_df = metadata_df[metadata_df['target_dataset'] == 'emime-eng']
    print("Total # utterances to convert: {}".format(len(selected_metadata_df)))

    for idx, row in tqdm(selected_metadata_df.iterrows(), total=len(selected_metadata_df)):
        source_dataset = row['source_dataset']
        source_speaker = row['source_speaker']
        source_utterance = row['source_utterance']

        source_audio_path = os.path.join(data_dir[source_dataset], source_utterance)
        assert os.path.exists(source_audio_path), f"File {source_audio_path} does not exist!"
        
        target_dataset = row['target_dataset']
        target_utterance = row['target_utterance']
        target_audio_path = os.path.join(data_dir[target_dataset], target_utterance)
        assert os.path.exists(target_audio_path), f"File {target_audio_path} does not exist!"

        out_wav_path = os.path.join(out_dir, os.path.basename(target_utterance).split('.')[0], "{}_{}".format(source_speaker, os.path.basename(source_utterance)))
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        if os.path.exists(out_wav_path):
            continue

        tgt_emb = speaker_embeddings[speaker_embeddings['target_utterance'] == target_utterance]['speaker_embedding'].values[0]
        tgt_emb = torch.from_numpy(tgt_emb).unsqueeze(0).to(device)

        if not args.streaming:
            synth_audio = inference_ns(source_audio_path, tgt_emb)
        else:
            synth_audio = inference(source_audio_path, tgt_emb, model_name=model_name, use_projector=use_projector, right_context=args.right_context)
        wavfile.write(out_wav_path, 16000, synth_audio)    