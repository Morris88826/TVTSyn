import os
import re
import glob
import tqdm
import torch
import librosa
import argparse
import pandas as pd
from NISQA.nisqa.NISQA_model import nisqaModel
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

# MOS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr = nemo_asr.models.EncDecCTCModelBPE.restore_from("asr/stt_en_conformer_ctc_xlarge_v1.10.0/stt_en_conformer_ctc_xlarge.tar.gz")

def normalize(text):
    # lowercase
    text = text.lower()
    # remove punctuation (anything not word or whitespace)
    text = re.sub(r"[^\w\s]", "", text)
    return text

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, choices=["slt24", "DarkStream", "DarkStream-logits", "DarkStream-bottleneck", "DarkStream-quant", "DarkStream-quant-TVT-80ms", "DarkStream-logits-TVT", "DarkStream-bottleneck-TVT", "DarkStream-bottleneck-TVT-ns", "DarkStream-quant-TVT", "DarkStream-quant-TVT-ns","GenVC-small", "GenVC-large"], help="Model type used for conversion")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="/home/mtseng/anony_benchmark/notebooks/evaluation_data.csv", help="Path to the metadata CSV file")
    parser.add_argument("--output_dir", type=str, default="/data/mtseng/ICLR2026", help="Directory where the converted audio files are saved")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(args.output_dir, 'results', args.model)
    output_dir = os.path.join(args.output_dir, 'results_quantitative', args.model)
    os.makedirs(output_dir, exist_ok=True)

    metadata_df = pd.read_csv(args.metadata)
    selected_metadata_df = metadata_df[metadata_df['target_dataset'] == 'emime-eng']


    result_path = os.path.join(output_dir, 'vc.csv')

    if not os.path.exists(result_path):
        result_df = []
        for idx, row in tqdm.tqdm(selected_metadata_df.iterrows(), total=len(selected_metadata_df)):
            source_dataset = row['source_dataset']
            source_speaker = row['source_speaker']
            source_utterance = row['source_utterance']
            target_utterance = row['target_utterance']

            audio_file = os.path.join(results_dir, os.path.basename(target_utterance).replace('.wav', ''), '{}_{}'.format(source_speaker, os.path.basename(source_utterance)))
            assert os.path.exists(audio_file), f"Converted audio not found: {audio_file}"

            gt_text_path = os.path.join(data_dir[source_dataset], source_utterance)
            if source_dataset in ['artic', 'l2artic']:
                gt_text_path = gt_text_path.replace('.wav', '.txt').replace('/wav/', '/transcript/')
            elif source_dataset == 'vctk':
                gt_text_path = gt_text_path.replace('.wav', '.txt').replace('/wav48/', '/txt/')
            assert os.path.exists(gt_text_path), f"GT text not found: {gt_text_path}"
            
            with open(gt_text_path, 'r') as f:
                gt_text = f.read().strip()
            gt_text = normalize(gt_text)

            # MOS
            args = {
                'mode': 'predict_file',
                'deg': audio_file,
                'pretrained_model': './NISQA/weights/nisqa.tar',
                'ms_channel': None,
                'output_dir': None
            }
            nisqa = nisqaModel(args, verbose=False)
            result = nisqa.predict()

            # # WER
            # pred_text = asr.transcribe([audio_file])[0]
            # wer = word_error_rate(hypotheses=[normalize(pred_text)], references=[normalize(gt_text)])

            result_df.append({
                'source_dataset': source_dataset,
                'source_speaker': source_speaker,
                'source_utterance': source_utterance,
                'target_dataset': row['target_dataset'],
                'target_utterance': target_utterance,
                'mos_pred': result['mos_pred'][0],
                # 'wer': wer
            })
        result_df = pd.DataFrame(result_df)
        result_df.to_csv(result_path, index=False)
    else:
        result_df = pd.read_csv(result_path)

    print("Overall MOS: {:.3f} ± {:.3f}".format(result_df['mos_pred'].mean(), result_df['mos_pred'].std()))
    # print("Overall WER: {:.3f} ± {:.3f}".format(result_df['wer'].mean(), result_df['wer'].std()))