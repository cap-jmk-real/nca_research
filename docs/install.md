# Install

Python 3.11 or newer is required. The CPU build of PyTorch is enough for the unit tests, the smoke run and the table recomputation. Training the full models is faster on a GPU.

<!-- ci:run -->
```bash
python -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt scikit-learn huggingface_hub scipy
```

`requirements.txt` lists `numpy` and `torch`. The benchmark runners in `stage0/` also import `scikit-learn` and `scipy`, and the BFCL split preparation downloads from the Hugging Face Hub.

## Data root

Training and the benchmark runners write large files to a directory given by the environment variable `NCA_DATA_ROOT`. The variable has no default.

<!-- ci:run -->
```bash
export NCA_DATA_ROOT="$PWD/.nca_data"
```

On the author's Mac the data root is on an external SSD. The rules for that setup are in [data-root.md](https://github.com/cap-jmk-real/nca_research/blob/main/docs/data-root.md).

## Build these docs

<!-- ci:run -->
```bash
pip install -r requirements-docs.txt
mkdocs build --strict
```

The site is not deployed anywhere. CI only checks that it builds.
