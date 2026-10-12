# Listening validation

The local CPU environment is `.venv-listening`. It contains PyTorch, ONNX
Runtime, SpeechBrain, Gradio, and a Jupyter kernel. It avoids the host GPU-driver
stall. The model is already exported under `artifacts/TVTSyn`.

## Review directly in VS Code (no web server)

Open [listening_validation.ipynb](../inference/notebooks/listening_validation.ipynb)
and select `.venv-listening/bin/python` as its kernel. Run the setup cells,
enter your name, and work through the six cases. Audio is embedded in notebook
outputs, so playback occurs on your computer even when Python runs over SSH;
no port forwarding is needed.

After each case, edit its `rate_case(...)` cell: set `result` to `"pass"`,
`"fail"`, or `"unsure"`, add timestamped `notes`, and run the cell. `None`
preserves an existing rating. The long case includes a focused 20.8–21.8 s
excerpt. At the end set `SAVE_REVIEW = True` and run the save cell to write
`artifacts/listening/listening-results.json`. You can save a partial review
and resume it later. Share this JSON when ready.

The notebook does not pretend to test the physical microphone or live demo
session lifecycle. Leave that field pending until you can complete the
separate microphone procedure below. The final notebook cell reports those
remaining checks honestly. Only the existing inference notebook runs models;
this review notebook plays the prepared comparison files.

## Start the prepared review

From the repository root:

```bash
.venv-listening/bin/python -m http.server 8765 --bind 127.0.0.1 --directory artifacts/listening
```

Open **http://localhost:8765**. If VS Code is connected over SSH, forward port
8765 in its Ports panel and open the forwarded address in your local browser.
The server serves only the listening pack. Stop it with Ctrl-C when finished.

The page includes source and target reference audio plus PyTorch whole,
PyTorch streaming, and ONNX streaming output for six cases. The long case
intentionally repeats the source; a click also audible in the repeated source
is not evidence of a conversion regression. Each case uses the same playback
gain for all its players. Unscaled float WAVs are retained beside the playback
WAVs. Numerical comparison status is displayed explicitly; investigate any
failed tolerance check before publication even if it sounds acceptable.

1. Use headphones and keep volume fixed. Compare the source with each output
   for intelligibility, target voice, and naturalness.
2. Compare **PyTorch streaming with ONNX streaming**, focusing on clicks,
   dropouts, abrupt changes, and noise near chunk boundaries.
3. Check the first phoneme and final consonants. Review short and partial-final
   cases for lost, duplicated, or padded tails. Whole-utterance output is a
   separate execution behavior and need not equal streaming output.
4. Mark each case Pass, Fail, or Needs investigation. Record timestamps and
   symptoms rather than only a global quality rating.
5. Complete the physical microphone check below, then click **Download
   listening-results.json**. Keep the downloaded JSON and share it for review;
   the webpage does not send it anywhere or publish anything.

Ratings are saved in browser local storage, keyed by the review's manifest.
The downloaded JSON records the model checkpoint and review ID so results are
not accidentally applied to a rebuilt pack. A fresh browser profile starts
with every case pending.

## Physical microphone test

FFmpeg and ffprobe must be on PATH. From the repository root:

```bash
TVTSYN_MODEL=artifacts/TVTSyn TVTSYN_DEVICE=cpu TVTSYN_OFFLINE=1 \
  .venv-listening/bin/python spaces/demo/app.py
```

Open the displayed local address (normally http://localhost:7860); forward
port 7860 as well when working over SSH. Use headphones to prevent feedback.

- Select ONNX and supply `artifacts/listening/target.npy` as the saved embedding,
  or choose `samples/target.wav` as reference audio. Click Prepare target.
- Record 5–10 seconds; confirm you hear converted output. Stop and check the
  final tail. Record again without re-preparing the target.
- Repeat for PyTorch. CPU PyTorch may process more slowly than the incoming
  audio; the demo bounds recording duration and playback buffering. This is
  not a real-time performance promise.
- Open a second browser profile/incognito window with
  `artifacts/listening/reverse_target.npy`; alternate recording in the two
  sessions and confirm neither session inherits the other's voice/state.
- Record the results and any delay or discontinuity in the review page.

The 100 ms model step, 80 ms lookahead, 200 ms fixed-scheduler startup input,
processing time, resampling, and browser/network buffering are separate
latency components. Judge artifacts separately from delayed but continuous
playback.

## Notebook

Open `inference/notebooks/inference.ipynb`. In VS Code choose **Select Kernel →
Python Environments → `.venv-listening/bin/python`** (TVTSyn listening, CPU).
The first cell prints the interpreter and checks that the package is installed.
Use Run All. A fresh environment may need internet access to download the two
pinned speaker encoders once; this machine already has them cached.

If recreating the environment elsewhere:

```bash
bash inference/setup_listening.sh
```

The setup script creates the environment and its kernel; it does not change a
pre-existing research environment or choose your IDE kernel automatically.

## Rebuild with these or your own recordings

```bash
.venv-listening/bin/python inference/prepare_listening.py --offline
# Or supply source/target WAVs (omit --offline if speaker models need downloading):
.venv-listening/bin/python inference/prepare_listening.py \
  --source path/to/source.wav --target path/to/target.wav \
  --output artifacts/listening-custom
```

Use a separate output directory for each review you want to retain. The tool
saves embeddings, all waveforms, manifest/provenance, and numerical metrics.
`--strict-parity` makes generation abort on an extended-tolerance failure; without it,
failures are recorded visibly for investigation and listening.

After reviewing, run:

```bash
.venv-listening/bin/python inference/check_listening.py path/to/listening-results.json
```

Publication follows a completed review, resolution of failed cases, and the
remaining authenticated release steps in [PUBLICATION.md](PUBLICATION.md).

## Numerical scope of this listening pack

The original two captured reference examples retain their original acceptance
criteria: whole-mode `atol=5e-5, rtol=1e-4`; streaming `atol=3e-4, rtol=1e-3`.
The reference validation script still enforces these values.

The expanded 30-second case, with the freshly extracted target speaker,
exceeded the stricter streaming reference criterion at 65 of 480,000 samples.
The maximum difference was 0.00111679, concentrated in output chunk 213; mean
absolute error was approximately 1e-6. A diagnostic export exposing quantizer
outputs found identical code indices throughout all 300 steps, with maximum
latent difference 4.77e-7 in the affected chunk. Disabling ONNX optimizations
left a similar discrepancy. This is consistent with decoder floating-point
sensitivity rather than a changed code selection or missing streaming state;
it is not a proof of identical perceived quality.

For these extended listening cases, the separate numerical criterion is
`atol=0.002, rtol=0.001`, **and** mean absolute error at most `0.000005`.
This is a measured acceptance criterion for this review set, not a guaranteed
bound for all recordings or hardware. The page displays both the stricter
reference result and this extended result. All six current cases meet the
extended criterion. Listen particularly around **21.2–21.4 seconds** in the
long example; perceived clicks or discontinuities still require investigation.
