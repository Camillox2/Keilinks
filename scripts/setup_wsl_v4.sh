#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == "Linux" ]] || { echo "Execute dentro do WSL2/Ubuntu." >&2; exit 1; }
python3 -m venv .venv-v4
source .venv-v4/bin/activate
python -m pip install --upgrade pip wheel setuptools
echo "Instale primeiro o PyTorch CUDA indicado em https://pytorch.org/get-started/locally/"
echo "Depois: pip install -r requirements-v4.txt"
echo "Mantenha o projeto em /home/... e não em /mnt/c para melhor I/O."
