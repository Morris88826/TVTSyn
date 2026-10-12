# Research evaluation

These scripts and cleared notebooks preserve the original evaluation workflows.
Install training dependencies and run from the repository root, for example
`python -m evaluation.evaluation --help`. Provide your own metadata CSV,
speaker-embedding pool, checkpoints, and output paths. Generated CSV/pickle data
is excluded from the release source; existing local files were preserved.

For supported inference usage, use the public API in `inference/`. Numerical
backend parity and local release checks are documented in `docs/RELEASE_VALIDATION.md`.

## Portable corpus locations and devices

Set `TVTSYN_DATA_ROOT` to your corpus parent directory (default `data/corpora`).
`evaluation/settings.py` lists the expected subdirectories. To use unrelated
locations, set `TVTSYN_DATASETS_JSON` to a JSON object mapping dataset keys
(`artic`, `l2artic`, `vctk`, `emime-eng`, `emime-fin`, `emime-ger`, `emime-man`,
`l2en`, `indictts`, `nptel`) to directories. Your private path configuration
should stay outside the Git checkout or in ignored local data.

Conversion entry points accept `--device cpu` (default), `--device cuda`, or
an explicit GPU device. Under `torchrun`, `--device cuda` selects the process's
local GPU rank. Set `TVTSYN_RESEARCH_DEVICE` to change the default across the
research scripts. `--data-root` overrides the corpus parent for conversion and
parallel-generation scripts. Run these scripts as modules from the repository
root so their shared settings can be imported.

These preserved research workflows still need their own datasets and optional
third-party systems such as NISQA or SoftVC; those systems are not dependencies
of the public inference API.
