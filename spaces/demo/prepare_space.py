"""Stage a reproducible Gradio Space; upload separately after inspecting output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel',required=True)
    p.add_argument('--revision',required=True,help='Immutable Hub model commit')
    p.add_argument('--source-revision',required=True,help='Git source commit used to build the wheel')
    p.add_argument('--hardware',choices=['cpu-basic','zero-a10g'],default='zero-a10g')
    p.add_argument('--output',default='artifacts/space')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    wheel=Path(a.wheel);shutil.copy2(wheel,out/wheel.name)
    app=(root/'spaces/demo/app.py').read_text()
    app=app.replace("REVISION = os.environ.get('TVTSYN_REVISION', '2d3147b8a465e1d9aad33a2a0e53956bf8ed32aa')",
                    'REVISION = os.environ.get("TVTSYN_REVISION", '+repr(a.revision)+')')
    (out/'app.py').write_text(app)
    shutil.copy2(root/'spaces/demo/packages.txt',out/'packages.txt')
    shutil.copy2(root/'LICENSE',out/'LICENSE')
    torch_requirements = ('torch==2.8.0\ntorchaudio==2.8.0\nspaces>=0.48,<1\n'
                          if a.hardware == 'zero-a10g' else
                          '--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.8.0+cpu\ntorchaudio==2.8.0+cpu\n')
    (out/'requirements.txt').write_text(torch_requirements+
        'huggingface-hub==0.36.2\ngradio==5.50.0\n./'+wheel.name+'[demo]\n')
    card=(root/'inference/tvtsyn_inference/model_card.md').read_text().split('---',2)[2].strip()
    front='''---
title: TVTSyn
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: '3.10'
app_file: app.py
license: cc0-1.0
models:
- warisqr007/TVTSyn
short_description: Voice conversion with PyTorch and ONNX streaming
---

'''
    (out/'README.md').write_text(front+'This Space loads the published model at revision `'+a.revision+'`.\n'
        'The app code retains CC0; TVTSyn weights use CC BY-NC 4.0.\n\n'+card+'\n')
    record={'model':'warisqr007/TVTSyn','model_revision':a.revision,'source_revision':a.source_revision,
            'hardware':a.hardware,'wheel':wheel.name,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()}
    (out/'release.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
