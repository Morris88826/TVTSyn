"""Bounded CPU resume validation using trusted local checkpoints and sample features.
Run: python -m training.scripts.validate_resume
"""
from pathlib import Path
import json,inspect
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from training.modules import PasrWithQuantModule,ResynthesisWithQuantAndTVTimbreModule
from training.data_modules.pasr import PasrDataset
from training.data_modules.resynthesis import ResynthesisDataset
root=Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
files=list((root/'data/speakers').glob('*/wav/*.wav'))*3
report=[]
for cls,ds,path in [(PasrWithQuantModule,PasrDataset,'mimi_pasrv2_withquant_la4.ckpt'),(ResynthesisWithQuantAndTVTimbreModule,ResynthesisDataset,'resynthesis_quant_and_tvtimbre_ft.ckpt')]:
 p=root/'ckpts'/path
 extra={'encoder_path':None} if cls is ResynthesisWithQuantAndTVTimbreModule else {}
 model=cls.load_from_checkpoint(p,map_location='cpu',weights_only=False,**extra)
 if cls is PasrWithQuantModule:
  # This optimizer is normally injected by LightningCLI from the stage-1 YAML.
  import types
  def configure(self):
   opt=torch.optim.AdamW(self.parameters(),lr=5e-4,weight_decay=1e-6)
   sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.1,patience=3,min_lr=1e-5)
   return {'optimizer':opt,'lr_scheduler':{'scheduler':sched,'monitor':'val/loss','strict':False}}
  model.configure_optimizers=types.MethodType(configure,model)
 checkpoint=torch.load(p,map_location='cpu',weights_only=False)
 step=checkpoint['global_step']; epoch=checkpoint['epoch']
 del checkpoint
 dataset=ds(files,segment_size=6400,code_hop_size=320,sampling_rate=16000)
 loader=DataLoader(dataset,batch_size=1,num_workers=0)
 trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,
  enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
  limit_val_batches=0,max_steps=step+2,max_epochs=-1,default_root_dir=str(root/'artifacts/training-smoke'))
 kwargs={'weights_only':False} if 'weights_only' in inspect.signature(trainer.fit).parameters else {}
 trainer.fit(model,train_dataloaders=loader,ckpt_path=p,**kwargs)
 assert trainer.global_step>step
 report.append({'stage':cls.__name__,'checkpoint_global_step':step,'resumed_global_step':trainer.global_step,'optimizer_count':len(trainer.optimizers)})
 print(report[-1],flush=True)
(root/'artifacts/training_validation.json').write_text(json.dumps(report,indent=2))
