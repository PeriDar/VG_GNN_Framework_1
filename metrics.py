"""
Metrics: classification & finance utilities (minimal).
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

def to_numpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def map9_to3(y9_pred):
    return (y9_pred // 3).astype(int)  # 0..8 -> first horizon state in {0,1,2}

def sharpe_ratio(returns):
    r = np.asarray(returns)
    if r.std() == 0: return 0.0
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)

def calmar_ratio(returns):
    eq = (1 + returns).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    mdd = dd.min()
    if mdd == 0.0:
        return 0.0
    return (1 + returns).prod() ** (252 / max(1, len(returns))) - 1.0 / abs(mdd + 1e-12)