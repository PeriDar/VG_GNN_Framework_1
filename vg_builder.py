"""
Visibility graph builder for Framework 1 (RVG/FVG + volume edges).
This is a minimal stub: fill the functions based on your thesis logic.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def _visible(i, j, t, y, eps=1e-12) -> bool:
    ti, tj, yi, yj = t[i], t[j], y[i], y[j]
    if tj == ti: 
        return False
    for k in range(i+1, j):
        tk, yk = t[k], y[k]
        y_lin = yi + (yj - yi) * ((tk - ti) / (tj - ti))
        if yk >= y_lin - eps:
            return False
    return True

def build_rvg_fvg_indices(values: np.ndarray, times: np.ndarray, use_fvg=True, max_span=None):
    n = len(values)
    r_edges, f_edges = [], []
    for i in range(n-1):
        for j in range(i+1, n):
            if max_span is not None and (j - i) > max_span: 
                break
            if _visible(i, j, times, values):
                if values[j] > values[i]:
                    r_edges.append((i, j))
                elif use_fvg and (values[j] < values[i]):
                    f_edges.append((i, j))
    return r_edges, f_edges

def build_volume_edges(n: int):
    return [(i, i+1) for i in range(n-1)]

def segment_mean(arr: np.ndarray, i: int, j: int) -> float:
    if j <= i+1: return float(arr[j])
    return float(arr[i+1:j].mean())

def compute_node_features(win: pd.DataFrame) -> np.ndarray:
    feats = [
        win['ret1'].values,
        win['ret5'].values,
        win['ret20'].values,
        win['vol_z'].values,
        win['rsi'].values,
        win['atr'].values,
        win['body_pct'].values,
        win['upper_wick_pct'].values,
        win['lower_wick_pct'].values,
    ]
    X = np.nan_to_num(np.vstack(feats).T, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def compute_edge_features(win: pd.DataFrame, edges, etype_id: int, price_key='close'):
    t = np.arange(len(win), dtype=float)
    y = win[price_key].values.astype(float)
    vol_z = win['vol_z'].values.astype(float)
    obv = win['obv'].values.astype(float)
    feats = []
    for (i, j) in edges:
        dt = float(j - i)
        slope = (y[j] - y[i]) / (dt + 1e-8)
        cum_ret = float(np.log(y[j] + 1e-8) - np.log(y[i] + 1e-8))
        vol_avg = segment_mean(vol_z, i, j)
        obv_delta = float(obv[j] - obv[i])
        w_cont = 0.5 * abs(cum_ret) + 0.5 * max(vol_avg, 0.0)
        onehot = [1.0 if k == etype_id else 0.0 for k in range(3)]  # 0=RVG,1=FVG,2=VOL
        feats.append(onehot + [slope, dt, cum_ret, vol_avg, obv_delta, w_cont])
    return np.array(feats, dtype=float)

def assemble_graph(win: pd.DataFrame, use_fvg=True, use_volume_edges=True, max_span=None, price_key='close'):
    y = win[price_key].values.astype(float)
    t = np.arange(len(win), dtype=float)
    r_edges, f_edges = build_rvg_fvg_indices(y, t, use_fvg=use_fvg, max_span=max_span)
    edges_all, eattr_all = [], []
    if r_edges:
        edges_all += r_edges
        eattr_all.append(compute_edge_features(win, r_edges, etype_id=0, price_key=price_key))
    if use_fvg and f_edges:
        edges_all += f_edges
        eattr_all.append(compute_edge_features(win, f_edges, etype_id=1, price_key=price_key))
    if use_volume_edges:
        v_edges = build_volume_edges(len(win))
        edges_all += v_edges
        eattr_all.append(compute_edge_features(win, v_edges, etype_id=2, price_key=price_key))
    if len(edges_all) == 0:
        edges_all = build_volume_edges(len(win))
        eattr_all.append(compute_edge_features(win, edges_all, etype_id=2, price_key=price_key))
    E = np.array(edges_all, dtype=int)
    E = np.vstack([E, E[:, ::-1]])  # undirected
    EA = np.vstack(eattr_all); EA = np.vstack([EA, EA])
    X = compute_node_features(win)
    return {"X": X, "E": E, "EA": EA}