"""
Utility helpers: seeding, dirs, experiment folder naming.
"""
import os, random, numpy as np, torch
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def exp_dir(root: str) -> str:
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"exp_{d}")
    ensure_dir(path)
    return path

