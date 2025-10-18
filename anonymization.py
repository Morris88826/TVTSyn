import os
import glob
import random
import torch
import librosa
import argparse
import pandas as pd
from scipy.io import wavfile
from IPython import display as disp
from pathlib import Path
from tqdm import tqdm
from src.modules import ResynthesisWithBottleneckAndTVTimbreModule, ResynthesisWithLogitsAndTVTimbreModule, ResynthesisWithQuantAndTVTimbreModule, SpeakerModule

os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Set the GPU device
MAX_WAV_VALUE = 32768.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = {
    # source datasets
    'src': "/data/mtseng/Voice-Privacy-Challenge-2024/data",
    # target datasets
    'emime-eng': "/data/mtseng/EMIME/UEDIN_bilingual_96kHz_data/English_talkers",
}


def inference_ns(source_audio_path, target_audio_path, speaker_encoder, sample_rate=16000):
    src_audio, sr = librosa.load(source_audio_path, sr=sample_rate)
    assert sr == sample_rate
    tgt_audio, sr = librosa.load(target_audio_path, sr=sample_rate)
    assert sr == sample_rate

    src_audio = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).to(device)
    tgt_audio = torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).to(device)
    tgt_embed = speaker_encoder(tgt_audio.squeeze(0))
    
    resynthesizer.eval()
    with torch.no_grad():
        resyth_audio = resynthesizer(src_audio, tgt_embed)
        resyth_audio = resyth_audio.squeeze()
        resyth_audio = resyth_audio * MAX_WAV_VALUE
        resyth_audio = resyth_audio.detach().cpu().numpy().astype('int16')
    return resyth_audio



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["DarkStream-logits-TVT", "DarkStream-bottleneck-TVT", "DarkStream-quant-TVT"], help="Model name to use")
    parser.add_argument("--out_dir", type=str, default="/data/mtseng/Voice-Privacy-Challenge-2024/outputs", help="Directory to save the converted audio files")
    args = parser.parse_args()


    speaker_encoder = SpeakerModule(device="cuda" if torch.cuda.is_available() else "cpu")
    use_projector = False
    if args.model == "DarkStream-logits-TVT":
        resynthesizer = ResynthesisWithLogitsAndTVTimbreModule.load_from_checkpoint(
            "./ckpts/resynthesis_with_logits_and_tvtimbre.ckpt",
            encoder_path=None,
            map_location="cuda"
        )
        use_projector = True
    elif args.model == "DarkStream-bottleneck-TVT":
        # Uncomment the following line to use the KM Resynthesis Module instead
        resynthesizer = ResynthesisWithBottleneckAndTVTimbreModule.load_from_checkpoint(
            "./ckpts/resynthesis_bottleneck_and_tvtimbre.ckpt",
            encoder_path=None,
            decoder_pretrained=None,
            kmeans_path=None,
            map_location="cuda"
        )
    elif args.model == "DarkStream-quant-TVT":
        resynthesizer = ResynthesisWithQuantAndTVTimbreModule.load_from_checkpoint(
            "./ckpts/resynthesis_quant_and_tvtimbre.ckpt",
            encoder_path=None,
            map_location="cuda"
        )

    resynthesizer = resynthesizer.eval()

    out_dir = "{}/{}".format(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    eval_sets = [
        "libri_dev_enrolls",
        "libri_dev_trials_m",
        "libri_dev_trials_f",
        "libri_test_enrolls",
        "libri_test_trials_m",
        "libri_test_trials_f",
        "train-clean-360",
    ]
    base_fpath = Path(data_dir['src'])

    for eval_set in eval_sets:
        files = open(f"{base_fpath}/{eval_set}/wav.scp", "r").readlines()
        print(f"Processing {eval_set} set")
        out_dir_set = os.path.join(out_dir, eval_set)
        os.makedirs(out_dir_set, exist_ok=True)
        summary = []
        for file in tqdm(files):
            if eval_set == "train-clean-360":
                file = file.strip().split(" ")[-2]
            else:
                file = file.strip().split(" ")[-1]
            source_audio_path = (base_fpath.parent / file)
            assert os.path.exists(source_audio_path), f"File {source_audio_path} does not exist!"

            target_audio_paths = glob.glob(f"{data_dir['emime-eng']}/**/*.wav", recursive=True)
            target_audio_path = random.choice(target_audio_paths)

            out_wav_path = os.path.join(out_dir_set, os.path.basename(file).split('.')[0] + ".wav")
            if os.path.exists(out_wav_path):
                continue
            synth_audio = inference_ns(source_audio_path, target_audio_path, speaker_encoder)
            wavfile.write(out_wav_path, 16000, synth_audio)

            summary.append({
                "source_audio_path": str(source_audio_path).replace(os.path.dirname(data_dir['src'])+'/', ""),
                "target_audio_path": str(target_audio_path).replace(data_dir['emime-eng'], "emime-eng"),
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(out_dir_set, "summary.csv"), index=False)