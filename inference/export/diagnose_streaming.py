"""Diagnostic for release-v1 quantizer indices and cross-backend float drift.
Run from the repository root after preparing artifacts/listening.
Writes artifacts/quantizer_diagnosis.json; does not change the release graph.
"""
import onnx,onnxruntime as ort,numpy as np,torch,soundfile as sf,json
from tvtsyn_inference import TVTSyn
from tvtsyn.utils.compile import no_compile
from pathlib import Path
torch.set_num_threads(4)
pt=TVTSyn.from_pretrained('artifacts/TVTSyn',backend='pytorch')
runtime=TVTSyn.from_pretrained('artifacts/TVTSyn',backend='onnx').runtime
m=onnx.load('artifacts/TVTSyn/onnx/tvtsyn_step_100ms.onnx')
names=['/quantizer/ArgMax_output_0','/quantizer/out_project/Conv_output_0']
for name,dtype in zip(names,[onnx.TensorProto.INT64,onnx.TensorProto.FLOAT]):
 m.graph.output.append(onnx.helper.make_tensor_value_info(name,dtype,None))
options=ort.SessionOptions();options.intra_op_num_threads=1
sess=ort.InferenceSession(m.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
del m
x,_=sf.read('artifacts/listening/long_source_float.wav',dtype='float32');speaker=np.load('artifacts/listening/target.npy')
ps=pt.runtime.initial_state();os=runtime.initial_state();captured=[]
pt.runtime.model.encoder.quantizer.register_forward_hook(lambda mod,args,out:captured.append((out[3].detach().numpy().copy(),out[0].detach().numpy().copy())))
report=[]
for i in range(0,len(x),1600):
 wav=x[i:i+1600];py,ps,ready=pt.runtime.step(wav,speaker,ps)
 feed={'wav_chunk':wav.reshape(1,1,1600),'speaker_embed':speaker}
 feed.update({f'state_{j:03d}':a for j,a in enumerate(os) if f'state_{j:03d}' in runtime.input_names})
 result=sess.run(runtime.output_names+names,feed); oy=result[0].reshape(-1);os=tuple(result[1:-2])
 pi,pz=captured.pop();oi,oz=result[-2:]
 delta=float(np.max(np.abs(py-oy)))
 if delta>0.0002 or not np.array_equal(pi.flatten(),oi.flatten()):
  item={'chunk':i//1600,'wave_max_abs':delta,'quantizer_codes_equal':bool(np.array_equal(pi.flatten(),oi.flatten())),'quantized_latent_max_abs':float(np.max(np.abs(pz-oz)))}
  print(item,flush=True);report.append(item)
Path('artifacts/quantizer_diagnosis.json').write_text(json.dumps(report,indent=2))
