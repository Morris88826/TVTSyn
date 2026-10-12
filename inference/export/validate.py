"""Validate a prepared bundle against captured references and measure CPU execution."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import soundfile as sf
import torch
from tvtsyn_inference import TVTSyn


def metrics(a, b):
    if a.shape != b.shape:
        raise AssertionError(f'Output length mismatch: {a.shape} != {b.shape}')
    d = np.abs(a-b)
    return {'max_abs': float(d.max()) if d.size else 0., 'mean_abs': float(d.mean()) if d.size else 0.}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--long-seconds', type=int, default=30)
    args = p.parse_args()
    torch.set_num_threads(4)
    ref = Path(args.reference)
    speaker = np.load(ref/'speaker.npy')
    models = {b: TVTSyn.from_pretrained(args.model, backend=b, local_files_only=True) for b in ['pytorch','onnx']}
    report = {'platform': platform.platform(), 'torch': torch.__version__,
              'pytorch_threads': 4, 'onnx_threads': 1, 'tolerances': {'whole_atol': 5e-5, 'streaming_atol': 3e-4, 'rtol': 1e-3},
              'examples': [], 'timing': {}}
    for path in sorted(ref.glob('source_*.npy')):
        i=path.stem.split('_')[-1]
        x=np.load(path)
        a=models['pytorch'].convert(x,speaker,mode='whole')
        original=np.load(ref/f'whole_{i}.npy').flatten()
        np.testing.assert_allclose(a,original,atol=5e-5,rtol=1e-4)
        y=models['pytorch'].convert(x,speaker,mode='streaming')
        z=models['onnx'].convert(x,speaker,mode='streaming')
        legacy=np.load(ref/f'stream_{i}.npy').flatten()
        np.testing.assert_allclose(y[:len(legacy)],legacy,atol=3e-4,rtol=1e-3)
        np.testing.assert_allclose(y,z,atol=3e-4,rtol=1e-3)
        report['examples'].append({'example':i, 'whole': metrics(a,original),
            'legacy_streaming_prefix': metrics(y[:len(legacy)],legacy), 'onnx': metrics(y,z),
            'input_samples':len(x),'output_samples':len(z)})
        for name,audio in [('pytorch_whole',a),('pytorch_stream',y),('onnx_stream',z)]:
            sf.write(ref/f'{name}_{i}.wav',audio,16000,subtype='FLOAT')
    # Exercise cache wraparound and measure a longer, real-speech stream.
    x=np.resize(np.load(ref/'source_0.npy'),args.long_seconds*16000)
    long_outputs={}
    for backend, model in models.items():
        session=model.stream(speaker)
        times=[]; pieces=[]; first_ready=None
        for offset in range(0,len(x),1600):
            start=time.perf_counter();chunk=session.push(x[offset:offset+1600]);elapsed=time.perf_counter()-start
            times.append(elapsed);pieces.append(chunk)
            if chunk.size and first_ready is None:first_ready=(offset+1600)/16
        pieces.append(session.flush());long_outputs[backend]=np.concatenate(pieces)
        steady=np.array(times[10:])
        report['timing'][backend]={'audio_seconds':args.long_seconds,'chunk_ms':100,
            'first_valid_output_after_input_ms':first_ready,'first_call_processing_ms':times[0]*1000,
            'steady_mean_processing_ms':float(steady.mean()*1000),'steady_p95_processing_ms':float(np.percentile(steady,95)*1000),
            'rtf':float(steady.mean()/.1)}
    np.testing.assert_allclose(long_outputs['pytorch'],long_outputs['onnx'],atol=3e-4,rtol=1e-3)
    report['long_stream_parity']=metrics(long_outputs['pytorch'],long_outputs['onnx'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
