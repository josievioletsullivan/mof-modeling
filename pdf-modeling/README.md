# PDF modeling — staged pipeline

CIFs -> PDFs -> mixtures -> decomposition -> scores. Each stage reads/writes
data/ so any step can be rerun independently.

## Setup
    conda env create -f environment.yml
    conda activate pdf
    export MP_API_KEY="your_key"   # https://materialsproject.org -> dashboard

## Run
    python pipeline/01_fetch_structures.py

Physics params + the canonical r-grid live in config.py (single source of truth).
Stretched-NMF runs in its own env: conda env create -f environment-snmf.yml
