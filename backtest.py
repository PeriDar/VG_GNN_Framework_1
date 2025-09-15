"""
Very simple directional backtest: long if UP, short if DOWN, flat otherwise.
"""
import numpy as np

def simple_directional_backtest(dir_pred_3cls, fwd_returns_1, cost_bps=1.0, slippage_bps=2.0):
    pos = (dir_pred_3cls == 2).astype(int) - (dir_pred_3cls == 0).astype(int)  # +1, 0, -1
    changes = np.abs(np.diff(np.r_[0, pos]))
    roundtrip_cost = (cost_bps + slippage_bps) / 10000.0
    ret_gross = pos * fwd_returns_1
    ret_net = ret_gross - changes * roundtrip_cost
    return ret_net