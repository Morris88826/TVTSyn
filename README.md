# TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization

### Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah and Ricardo Gutierrez-Osuna


**Abstract :**
Real-time voice conversion and speaker anonymization require causal, low-latency synthesis without sacrificing intelligibility or naturalness. Current systems have a core representational mismatch: content is time-varying, while speaker identity is injected as a static global embedding. We introduce a streamable speech synthesizer that aligns the temporal granularity of identity and content via a content-synchronous, time-varying timbre (TVT) representation. A Global Timbre Memory expands a global timbre instance into multiple compact facets; frame-level content attends to this memory, a gate regulates variation, and spherical interpolation preserves identity geometry while enabling smooth local changes. In addition, a factorized vector-quantized bottleneck regularizes content to reduce residual speaker leakage. The resulting system is streamable end-to-end, with $<$80 ms GPU latency. Experiments show improvements in naturalness, speaker transfer, and anonymization  compared to SOTA streaming baselines, establishing TVT as a scalable approach for privacy-preserving and expressive speech synthesis under strict latency budgets.

Visit our [demo website](https://anonymized0826.github.io/TVTSyn/) for audio samples.
See submitted paper for additional details [here](TVTSyn_ICLR26.pdf)


## Usage

### Installation

```bash
# Clone project
git clone https://github.com/PSI-TAMU/TVTSyn.git 
cd TVTSyn

# [Optional] Create a conda virtual environment
conda create -n <env_name> python=3.10
conda activate <env_name>

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing
We follow the same data-processing stage as [here](https://github.com/PSI-TAMU/streamVC). Please follow directions in the specified repo.


### Run

**Fit**

```bash
python src/main.py fit -c configs/data/pasr.yaml -c configs/model/pasrwithquant_la4.yaml --trainer.logger.name exp1
```

**Resume**

```bash
python src/main.py fit -c configs/data/pasr.yaml -c configs/model/pasrwithquant_la4.yaml --ckpt_path <ckpt_path> --trainer.logger.id exp1_id
```

The model is trained in 2 steps:
1. Train the content encoder. For this use [pasr.yaml](configs/data/pasr.yaml) for data config and [pasrwithquant_la4.yaml](configs/model/pasrwithquant_la4.yaml) for model configs. `la4` corresponds to lookahead size of 80ms (i.e., 4 * 20ms frames). Change `model.init_args.transformer_args.right_context` accordingly to adjust lookahead size.

2. Training resynthesis module. For this use [resynthesis.yaml](configs/data/resynthesis.yaml) for data config and [resynthesisquantandtvt_la4.yaml](configs/model/resynthesisquantandtvt_la4.yaml) for model configs. Set `model.init_args.encoder_path` to the trained checkpoint from step 1.


## Checkpoints

| Model type                                    |  Link    |
|-----------------------------------------------|----------|
| Content Encoder  (From Step 1)                | [GDrive](https://drive.google.com/file/d/1_RAPZyWNkHdWuIVcKuukYrNczPG-PwXu/view?usp=sharing) |
| Content Encoder + Synthesizer (After Step 2)  | [GDrive](https://drive.google.com/file/d/120UQ0cdhZnYaQUxQX6hGLL2FN4rShPkA/view?usp=sharing) |


### Inference

1. See [inference.ipynb](notebooks/inference.ipynb)



