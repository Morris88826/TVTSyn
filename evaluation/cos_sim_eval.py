from evaluation.settings import dataset_roots, resolve_device
import os
import sys
import re
import glob
import tqdm
import torch
import librosa
import pandas as pd
from training.modules import SpeakerModule


speaker_encoder = SpeakerModule(device=os.environ.get("TVTSYN_RESEARCH_DEVICE", "cpu"))
model_name = "DarkStream-quant64-TVT-mel-voc-ns"

data_dir = dataset_roots()
results_dir = "ICLR26Results/{}".format(model_name)
output_dir = "ICLR26Results_quantitative/{}".format(model_name)
os.makedirs(output_dir, exist_ok=True)

metadata_path = "evaluation/evaluation_data.csv"
metadata_df = pd.read_csv(metadata_path)

selected_metadata_df = metadata_df[metadata_df['target_dataset'] == 'emime-eng']
print("Total # utterances to convert: {}".format(len(selected_metadata_df)))


def compute_cosine_similarity(audio_file1, audio_file2, sr=16000):
    audio1, _ = librosa.load(audio_file1, sr=sr)
    audio2, _ = librosa.load(audio_file2, sr=sr)
    
    audio1_em  = torch.from_numpy(audio1).unsqueeze(0).to(os.environ.get("TVTSYN_RESEARCH_DEVICE", "cpu"))
    audio2_em  = torch.from_numpy(audio2).unsqueeze(0).to(os.environ.get("TVTSYN_RESEARCH_DEVICE", "cpu"))

    embed1 = speaker_encoder(audio1_em)
    embed2 = speaker_encoder(audio2_em)

    # the resulting embeddings can be used for cosine similarity-based retrieval
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(embed1, embed2)
    return similarity.item()


result_path = os.path.join(output_dir, 'xv_ecapa_cosine_sim.csv')

if not os.path.exists(result_path):
    result_df = []
    for idx, row in tqdm.tqdm(selected_metadata_df.iterrows(), total=len(selected_metadata_df)):
        source_utterance = os.path.join(data_dir[row['source_dataset']], row['source_utterance'])
        assert os.path.exists(source_utterance), f"Source utterance not found: {source_utterance}"
        target_utterance = os.path.join(data_dir[row['target_dataset']], row['target_utterance'])
        assert os.path.exists(target_utterance), f"Target utterance not found: {target_utterance}"

        vc_utterance = os.path.join(results_dir, os.path.basename(row['target_utterance']).replace('.wav', ''), '{}_{}'.format(row['source_speaker'], os.path.basename(row['source_utterance'])))
        assert os.path.exists(vc_utterance), f"VC utterance not found: {vc_utterance}"

        src_cos_sim = compute_cosine_similarity(source_utterance, vc_utterance)
        tgt_cos_sim = compute_cosine_similarity(target_utterance, vc_utterance)

        result_df.append({
            'source_dataset': row['source_dataset'],
            'source_speaker': row['source_speaker'],
            'source_utterance': source_utterance,
            'target_dataset': row['target_dataset'],
            'target_utterance': target_utterance,
            'src_cos_sim': src_cos_sim,
            'tgt_cos_sim': tgt_cos_sim,
        })

    result_df = pd.DataFrame(result_df)
    result_df.to_csv(result_path, index=False)
else:
    result_df = pd.read_csv(result_path)


print("Average source cosine similarity: {:.3f} +/- {:.2f}".format(result_df['src_cos_sim'].mean(), result_df['src_cos_sim'].std()))
print("Average target cosine similarity: {:.3f} +/- {:.2f}".format(result_df['tgt_cos_sim'].mean(), result_df['tgt_cos_sim'].std()))
