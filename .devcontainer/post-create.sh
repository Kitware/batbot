#!/usr/bin/env bash
set -euo pipefail

git lfs install --local
git lfs pull
python -m pip install --upgrade pip
python -m pip install -e '.[cpu,test,docs,notebook,dev]'
python -m pip check
python -m ipykernel install --user --name batbot --display-name 'Python (BatBot)'
mkdir -p data/input artifacts/notebooks
printf '%s\n' 'Ready: open examples/notebooks/01_classify_your_files.ipynb and select Python (BatBot).'
