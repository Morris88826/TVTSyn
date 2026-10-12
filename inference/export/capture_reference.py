import os, sys, json, hashlib, inspect, importlib.util
from pathlib import Path
os.environ['TORCH_COMPILE_DISABLE']='1'
import argparse
parser=argparse.ArgumentParser(description='Capture baseline outputs before a model refactor')
parser.add_argument('--legacy-root', required=True)
parser.add_argument('--repository', default='.')
parser.add_argument('--checkpoint', default='ckpts/resynthesis_quant_and_tvtimbre_ft.ckpt')
parser.add_argument('--qt-root', required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(args.legacy_root).resolve()))
import torch, numpy as np, soundfile as sf
from src.modules import ResynthesisWithQuantAndTVTimbreModule
from src.components import data_classes as dc
root=Path(args.repository).resolve(); out=root/'artifacts/reference'; out.mkdir(parents=True,exist_ok=True)
p=root/args.checkpoint
checkpoint=torch.load(p,map_location='cpu',weights_only=False)
hp=dict(checkpoint['hyper_parameters'])
(out/'checkpoint_config.json').write_text(json.dumps(hp,indent=2))
kwargs={k:v for k,v in hp.items() if k in inspect.signature(ResynthesisWithQuantAndTVTimbreModule).parameters}
for key, cls in [('seanet_args',dc.SeanetArgs),('transformer_args',dc.TransformerArgs),('quantizer_args',dc.FVQuantizerArgs),('decoder_seanet_args',dc.DecSeanetNCArgs),('frame_synthesizer_args',dc.FrameSynthesizerArgs),('tvtimbre_args',dc.TVTimbreArgs),('lambdas',dc.Lambdas)]:
 if key in kwargs: kwargs[key]=cls(**kwargs[key])
kwargs['encoder_path']=None
model=ResynthesisWithQuantAndTVTimbreModule(**kwargs).eval()
result=model.load_state_dict(checkpoint['state_dict'],strict=False)
assert not [k for k in result.missing_keys if k.startswith(('encoder.net.','decoder.'))],result
# Use real saved speaker embeddings bundled with the training examples.
embed=np.load(root/'data/speakers/14/spkr/14_208_000001_000000.spk.npy').reshape(1,-1).astype('float32')
assert embed.shape==(1,704),embed.shape
np.save(out/'speaker.npy',embed)
spec=importlib.util.spec_from_file_location('reference_parity',str(Path(args.qt_root)/'tvtsyn/streaming/parity.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(4)
results=[]
with torch.inference_mode():
 for idx,name in enumerate(['sample_wav/BDL/arctic_a0001.wav','sample_wav/SLT/arctic_a0001.wav']):
  wav,sr=sf.read(root/name,dtype='float32'); assert sr==16000
  wav=wav[:16000*3]; np.save(out/f'source_{idx}.npy',wav)
  x=torch.from_numpy(wav)[None,None,:]; s=torch.from_numpy(embed)
  whole=model(x,s).numpy(); stream=m.run_legacy_streaming_reference(model,x,s,1600).numpy()
  np.save(out/f'whole_{idx}.npy',whole);np.save(out/f'stream_{idx}.npy',stream)
  results.append({'source':name,'samples':len(wav),'whole_shape':list(whole.shape),'stream_shape':list(stream.shape)})
digest=hashlib.sha256()
with p.open('rb') as handle:
 for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
manifest={'checkpoint':p.name,'sha256':digest.hexdigest(),'torch':torch.__version__,'examples':results,'loading':{'missing':result.missing_keys,'unexpected':result.unexpected_keys},'reference_note':'Legacy implementation snapshot; streaming reference retains original notebook tail behavior.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
