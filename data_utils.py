"""
Data loading, indicators, labels and temporal splits.
(Fill/extend functions as needed.)
"""
import pandas as pd
import numpy as np

def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        df = df.sort_values(df.columns[0]).reset_index(drop=True)
    required = ['open','high','low','close','volume']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    return df

def add_indicators(df: pd.DataFrame, rsi_period=14, atr_period=14) -> pd.DataFrame:
    out = df.copy()
    out['ret1'] = np.log(out['close']).diff()
    out['ret5'] = np.log(out['close']).diff(5)
    out['ret20'] = np.log(out['close']).diff(20)
    rng = (out['high'] - out['low']).replace(0, np.nan)
    body = (out['close'] - out['open']).abs()
    out['range_pct'] = (out['high'] - out['low']) / (out['close'].shift(1).abs() + 1e-8)
    out['body_pct'] = (body / (rng + 1e-8)).fillna(0.0)
    upper = (out['high'] - out[['open','close']].max(axis=1))
    lower = (out[['open','close']].min(axis=1) - out['low'])
    out['upper_wick_pct'] = (upper / (rng + 1e-8)).fillna(0.0)
    out['lower_wick_pct'] = (lower / (rng + 1e-8)).fillna(0.0)
    delta = out['close'].diff()
    up = delta.clip(lower=0.0); dn = -delta.clip(upper=0.0)
    roll_up = up.rolling(rsi_period, min_periods=1).mean()
    roll_dn = dn.rolling(rsi_period, min_periods=1).mean()
    rs = roll_up / (roll_dn + 1e-12)
    out['rsi'] = 100 - (100 / (1 + rs))
    hl = out['high'] - out['low']
    hc = (out['high'] - out['close'].shift(1)).abs()
    lc = (out['low'] - out['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out['atr'] = tr.rolling(atr_period, min_periods=1).mean()
    sign = np.sign(out['close'].diff().fillna(0.0))
    out['obv'] = (sign * out['volume']).fillna(0.0).cumsum()
    vol_roll = out['volume'].rolling(100, min_periods=20)
    out['vol_z'] = (out['volume'] - vol_roll.mean()) / (vol_roll.std() + 1e-12)
    return out

def label_9class(df: pd.DataFrame, horizons=(1,5), flat_bps=5):
    thr = flat_bps / 10000.0
    r1 = np.log(df['close']).shift(-horizons[0]) - np.log(df['close'])
    r2 = np.log(df['close']).shift(-horizons[1]) - np.log(df['close'])
    def tri(ret): 
        return np.where(ret > thr, 2, np.where(ret < -thr, 0, 1))
    s1, s2 = tri(r1), tri(r2)
    return pd.Series(s1 * 3 + s2, index=df.index)

def label_return(df: pd.DataFrame, horizon=1):
    return np.log(df['close']).shift(-horizon) - np.log(df['close'])

def make_splits(df: pd.DataFrame, train_end, val_end):
    if 'timestamp' not in df.columns:
        # fallback to index slicing (customize as needed)
        n = len(df)
        te = int(0.7 * n); ve = int(0.85 * n)
        return {"train": df.index[:te], "val": df.index[te:ve], "test": df.index[ve:]}
    train_idx = df.index[df['timestamp'] <= pd.to_datetime(train_end)]
    val_idx = df.index[(df['timestamp'] > pd.to_datetime(train_end)) & (df['timestamp'] <= pd.to_datetime(val_end))]
    test_idx = df.index[df['timestamp'] > pd.to_datetime(val_end)]
    return {"train": train_idx, "val": val_idx, "test": test_idx}