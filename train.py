"""
Training script (skeleton): loads data, builds graphs, trains model, evaluates.
Fill in as needed; compatible with configs in configs/.
"""
import argparse, os, yaml, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from utils import set_seed, exp_dir
from data_utils import load_ohlcv, add_indicators, label_9class, label_return, make_splits
from dataset import WindowGraphDataset
from models import EdgeAwareECC, GINBaseline
from metrics import acc, macro_f1, map9_to3, sharpe_ratio
from backtest import simple_directional_backtest

def load_config(path): 
    with open(path,'r') as f: 
        return yaml.safe_load(f)

def class_weights_from_loader(loader, num_classes=9):
    counts = np.zeros(num_classes, dtype=float)
    for batch in loader:
        y = batch.y_cls9.numpy()
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)

def build_model(cfg, in_dim, edge_dim):
    if cfg['model'] == 'edge_ecc':
        return EdgeAwareECC(in_dim, edge_dim, hidden=cfg['hidden_dim'], num_layers=cfg['num_layers'], readout=cfg['readout'], num_classes=9)
    elif cfg['model'] == 'gin':
        return GINBaseline(in_dim, hidden=cfg['hidden_dim'], num_layers=cfg['num_layers'], readout=cfg['readout'], num_classes=9)
    else:
        raise ValueError("Unknown model")

def evaluate(model, loader, device):
    model.eval()
    ys, ps, rs, r_hat = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, rpred = model(batch)
            pred = logits.argmax(dim=1)
            ys.append(batch.y_cls9.cpu().numpy())
            ps.append(pred.cpu().numpy())
            rs.append(batch.y_ret.cpu().numpy())
            r_hat.append(rpred.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    r = np.concatenate(rs).reshape(-1); rhat = np.concatenate(r_hat).reshape(-1)
    a = acc(y, p); f = macro_f1(y, p)
    dir3 = map9_to3(p)
    ret = simple_directional_backtest(dir3, r)
    sr = sharpe_ratio(ret)
    return {"acc": a, "f1": f, "sharpe": sr, "ret_series": ret, "y": y, "p": p, "r": r, "rhat": rhat}

def main(cfg):
    set_seed(cfg['seed'])
    out_dir = exp_dir(cfg['out_dir'])
    # Data
    df = load_ohlcv(cfg['csv_path'])
    df = add_indicators(df, rsi_period=cfg['rsi_period'], atr_period=cfg['atr_period'])
    y9 = label_9class(df, horizons=tuple(cfg['horizons']), flat_bps=cfg['flat_threshold_bps'])
    yret = label_return(df, horizon=cfg['horizons'][0])
    splits = make_splits(df, cfg['train_end'], cfg['val_end'])
    graph_kwargs = dict(use_fvg=cfg['use_fvg'], use_volume_edges=cfg['use_volume_edges'], max_span=cfg['max_vis_span'], price_key='close')
    ds_train = WindowGraphDataset(df, splits['train'], cfg['window_length'], cfg['step'], y9, yret, graph_kwargs, cfg['use_structural_features'])
    ds_val   = WindowGraphDataset(df, splits['val'],   cfg['window_length'], cfg['step'], y9, yret, graph_kwargs, cfg['use_structural_features'])
    ds_test  = WindowGraphDataset(df, splits['test'],  cfg['window_length'], cfg['step'], y9, yret, graph_kwargs, cfg['use_structural_features'])
    train_loader = DataLoader(ds_train, batch_size=cfg['batch_size'], shuffle=False)
    val_loader   = DataLoader(ds_val,   batch_size=cfg['batch_size'], shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=cfg['batch_size'], shuffle=False)
    in_dim = ds_train[0].x.size(1); edge_dim = ds_train[0].edge_attr.size(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, in_dim, edge_dim).to(device)
    ce = CrossEntropyLoss(weight=class_weights_from_loader(train_loader).to(device)) if cfg['class_weighting'] else CrossEntropyLoss()
    mse = MSELoss()
    opt = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    best_f1, patience, no_imp = -1.0, cfg['patience'], 0
    best_path = os.path.join(out_dir, 'best.pt')
    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train(); loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            logits, rpred = model(batch)
            loss_cls = ce(logits, batch.y_cls9)
            loss_reg = mse(rpred.view(-1), batch.y_ret.view(-1))
            loss = loss_cls + cfg['lambda_reg'] * loss_reg
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item())
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | TrainLoss {loss_sum:.4f} | ValF1 {val_metrics['f1']:.4f} | ValAcc {val_metrics['acc']:.4f} | ValSharpe {val_metrics['sharpe']:.3f}")
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']; torch.save(model.state_dict(), best_path); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST | Acc {test_metrics['acc']:.4f} | F1 {test_metrics['f1']:.4f} | Sharpe {test_metrics['sharpe']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)