#!/usr/bin/env bash
# Run from any directory. Set TVTSYN_PYTHON to choose a Python 3.10+ interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
"${TVTSYN_PYTHON:-python3}" -m venv .venv-listening
.venv-listening/bin/python -m pip install 'torch==2.8.0+cpu' 'torchaudio==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv-listening/bin/python -m pip install -e '.[demo]' ipykernel nbformat nbclient
.venv-listening/bin/python -m ipykernel install --prefix .venv-listening --name tvtsyn-listening --display-name 'TVTSyn listening (CPU)'
printf '%s\n' 'Ready. In VS Code select .venv-listening/bin/python as the notebook interpreter.'
