"""Build an offline listening review using the public inference API."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def render_review(manifest):
    data = json.dumps(manifest).replace('<', '\\u003c')
    rows = []
    for case in manifest['cases']:
        players = ''.join(f'<div><b>{html.escape(label)}</b><audio controls preload="none" src="{html.escape(path)}"></audio></div>'
                          for label, path in case['audio'].items())
        key = html.escape(case['id'])
        rows.append(f'''<section><h2>{html.escape(case['title'])}</h2><p>{html.escape(case['purpose'])}</p>
<p>Extended listening tolerance passed: {case['release_tolerance_passed']}. Original reference tolerance (stricter): {case['reference_tolerance_passed']}. Max absolute difference: {case['streaming_max_abs']:.6g}.</p>
<p>Shared playback gain: {case['playback_gain']:.4f}. No per-output normalization.</p>
<div class="players">{players}</div>
<label>Result <select data-result="{key}"><option value="pending">Not reviewed</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="unsure">Needs investigation</option></select></label>
<label>Notes / timestamps <textarea data-notes="{key}" placeholder="Clicks near 1.2 s, missing final consonant, target similarity, etc."></textarea></label></section>''')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TVTSyn listening validation</title><style>
body{font:17px/1.5 system-ui;background:#f3f5f7;color:#182c3d;max-width:1120px;margin:32px auto;padding:0 20px}
section,header{background:white;padding:24px;border-radius:12px;margin-bottom:20px;border:1px solid #dce3e9}
h1{margin-top:0}h2{font-size:22px}.players{display:flex;flex-wrap:wrap;gap:18px}.players div{flex:1 1 300px}
audio{display:block;width:100%;margin:10px 0 22px}label{display:block;margin:12px 0}textarea{display:block;width:95%;min-height:70px}
select,input,button,textarea{font:inherit;padding:8px}button{background:#143c52;color:white;border:0;border-radius:6px;cursor:pointer}
#status{font-weight:bold}small{color:#476071}</style>
<header><h1>TVTSyn listening validation</h1><p>Use headphones at a comfortable, fixed volume. Review intelligibility, target voice, clicks near 100 ms boundaries, first sounds, and the final tail. PyTorch whole and streaming are distinct behaviors; compare PyTorch streaming most closely with ONNX streaming.</p>
<p>Playback files use a shared gain within each case to avoid clipping. Original float outputs and numerical metrics are saved alongside this page. Tiny clips test draining; they may be too short to judge voice quality.</p>
<label>Reviewer <input id="reviewer" placeholder="Name or initials"></label>
<label>Headphones / browser / observations <input id="environment" size="48"></label>
<p id="status"></p><small>Results stay in this browser until you download the JSON. Rebuilding a different review creates a separate results record.</small></header>
''' + ''.join(rows) + '''<section><h2>Physical microphone check</h2><p>In the local Gradio demo: prepare a target, record speech, listen, stop and listen to the final tail, then record again without preparing. Repeat with both backends. Use a second browser session with another target to check isolation.</p>
<label>Microphone / stop / restart / isolation <select id="microphone"><option value="pending">Not tested</option><option value="pass">Passed both backends</option><option value="fail">Failed / needs investigation</option></select></label>
<label>Microphone notes <textarea id="mic-notes"></textarea></label>
<button id="download">Download listening-results.json</button><p>Send the JSON back for release review. Nothing is published by this page.</p></section>
<script>
const manifest = ''' + data + ''';
const key = 'tvtsyn-listening-' + manifest.review_id;
const controls = [...document.querySelectorAll('input,select,textarea')];
function id(e){return e.id || (e.dataset.result ? 'result:' + e.dataset.result : 'notes:' + e.dataset.notes)}
function collect(){const values={};for(const e of controls)values[id(e)]=e.value;return values}
function update(){const values=collect();localStorage.setItem(key,JSON.stringify(values));const done=[...document.querySelectorAll('[data-result]')].filter(e=>e.value==='pass').length;document.querySelector('#status').textContent=done+' / '+manifest.cases.length+' cases marked pass.'}
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
for(const e of controls){if(saved[id(e)])e.value=saved[id(e)];e.addEventListener('input',update)}
update();
document.querySelector('#download').onclick=()=>{const values=collect();const results={review_id:manifest.review_id,model:manifest.model,checkpoint_sha256:manifest.checkpoint_sha256,reviewed_at:new Date().toISOString(),reviewer:values.reviewer,environment:values.environment,cases:manifest.cases.map(c=>({id:c.id,result:values['result:'+c.id],notes:values['notes:'+c.id]})),microphone:{result:values.microphone,notes:values['mic-notes']}};const url=URL.createObjectURL(new Blob([JSON.stringify(results,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='listening-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
// Avoid overlapping comparisons.
for(const a of document.querySelectorAll('audio'))a.addEventListener('play',()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()});
</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='artifacts/TVTSyn')
    p.add_argument('--source', default='samples/source.wav')
    p.add_argument('--target', default='samples/target.wav')
    p.add_argument('--output', default='artifacts/listening')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--strict-parity', action='store_true', help='Abort if a case exceeds the documented extended listening tolerance')
    a = p.parse_args()
    import torch
    from tvtsyn_inference import TVTSyn
    from tvtsyn_inference.audio import load_audio, resample
    torch.set_num_threads(a.threads)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    pt = TVTSyn.from_pretrained(a.model, backend='pytorch', device='cpu', local_files_only=a.offline)
    ort = TVTSyn.from_pretrained(a.model, backend='onnx', local_files_only=a.offline)
    src, sr = load_audio(a.source); src = resample(src, sr)
    tgt, sr = load_audio(a.target); tgt = resample(tgt, sr)
    # Seed extraction for repeatability; saved embeddings remain the reference.
    torch.manual_seed(17)
    speaker = pt.encode_speaker(tgt)
    torch.manual_seed(17)
    reverse_speaker = pt.encode_speaker(src)
    np.save(out/'target.npy', speaker); np.save(out/'reverse_target.npy', reverse_speaker)
    cases = [
        ('forward', 'Source → target', src, tgt, speaker, 'Compare intelligibility, target identity, onset and final consonants.'),
        ('reverse', 'Reverse direction', tgt, src, reverse_speaker, 'Check conversion in the other speaker direction.'),
        ('short', 'Short clip: 80 ms', src[len(src)//2:len(src)//2+1280], tgt, speaker, 'Check short-input draining and clicks. Voice identity is not meaningful on this fragment.'),
        ('partial', 'Partial final chunk', src[:min(len(src),33617)], tgt, speaker, 'Listen to the end: no dropped or repeated tail after a partial 100 ms chunk.'),
        ('silence', 'Silence around speech', np.pad(src,(1600,2400)), tgt, speaker, 'Check transitions into speech and back to silence; listen for spurious bursts.'),
        ('long', '30-second stream', np.resize(src,480000), tgt, speaker, 'Check cache wraparound and stability. Source repeats intentionally; compare source repeat boundaries before attributing clicks to conversion.'),
    ]
    manifest = {'format_version':1,'model':a.model,'checkpoint_sha256':pt.config['provenance']['checkpoint_sha256'],
                'extended_tolerance':{'atol':2e-3,'rtol':1e-3,'max_mean_abs':5e-6},'config_sha256':digest(pt.root/'config.json'),'source_sha256':digest(a.source),'target_sha256':digest(a.target),
                'speaker_sha256':digest(out/'target.npy'),'sample_rate':16000,'cases':[]}
    for key,title,x,target,s,purpose in cases:
        print('Generating',key,flush=True)
        waves={'Source':x,'Target reference':target,'PyTorch whole':pt.convert(x,s,mode='whole'),
               'PyTorch streaming':pt.convert(x,s,mode='streaming'),'ONNX streaming':ort.convert(x,s)}
        parity_ok = bool(np.allclose(waves['PyTorch streaming'],waves['ONNX streaming'],atol=3e-4,rtol=1e-3))
        delta = np.abs(waves['PyTorch streaming'] - waves['ONNX streaming'])
        release_ok = bool(np.allclose(waves['PyTorch streaming'], waves['ONNX streaming'], atol=2e-3, rtol=1e-3) and delta.mean() <= 5e-6)
        if a.strict_parity and not release_ok:
            raise AssertionError(f'{key}: extended listening tolerance exceeded')
        # Check callback-size independence for a real speech clip.
        if key=='forward':
            session=ort.stream(s)
            try:
                irregular=np.concatenate([session.push(x[i:i+731]) for i in range(0,len(x),731)]+[session.flush()])
            finally: session.close()
            np.testing.assert_array_equal(irregular,waves['ONNX streaming'])
        gain=min(1.0,.95/max(float(np.max(np.abs(w))) for w in waves.values()))
        files={}
        for label,w in waves.items():
            name=key+'_'+label.lower().replace(' ','_')
            sf.write(out/(name+'_float.wav'),w,16000,subtype='FLOAT')
            sf.write(out/(name+'.wav'),w*gain,16000,subtype='PCM_16')
            files[label]=name+'.wav'
        delta=np.abs(waves['PyTorch streaming']-waves['ONNX streaming'])
        manifest['cases'].append({'id':key,'title':title,'purpose':purpose,'input_samples':len(x),
            'playback_gain':gain,'audio':files,'reference_tolerance_passed':parity_ok,'release_tolerance_passed':release_ok,'streaming_max_abs':float(delta.max()),
            'streaming_mean_abs':float(delta.mean()),'output_samples':{label:len(w) for label,w in waves.items()}})
    manifest['review_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'index.html').write_text(render_review(manifest))
    print('Ready:',out/'index.html')


if __name__ == '__main__':
    main()
