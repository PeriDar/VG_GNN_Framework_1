# GNN + Visibility Graph Framework (Framework 1 Skeleton)

This repository is a clean starter skeleton for your thesis experiments on **Multiplex Directional Visibility Graphs (VG) + GNN**.

## Structure
```
gnn-vg-framework/
├── configs/
│   ├── config.yaml
│   └── config_debug.yaml
├── data/                 # put your CSVs here (gitignored)
├── runs/                 # outputs/checkpoints (gitignored)
├── backtest.py
├── data_utils.py
├── dataset.py
├── metrics.py
├── models.py
├── train.py
├── utils.py
├── vg_builder.py
├── requirements.txt
└── .gitignore
```

## Quickstart (local)
1. Create and activate a Python environment (conda recommended).
2. Install requirements: `pip install -r requirements.txt` (see notes below for PyTorch Geometric).
3. Place your OHLCV CSV at `data/ASSET.csv` (columns: `timestamp,open,high,low,close,volume`).
4. Edit `configs/config.yaml` and run:
   ```bash
   python train.py --config configs/config.yaml
   ```

## Notes on PyTorch Geometric (PyG)
PyG wheels depend on your PyTorch & CUDA versions. See official: https://pytorch-geometric.readthedocs.io
On Colab you can install in cells; locally use compatible wheels for your setup.

## Git workflow (reminder)
```bash
git init
git add .
git commit -m "Initial commit"
# create empty repo on GitHub, then:
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```