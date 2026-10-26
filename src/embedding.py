"""Turns an account's SQL features into a fixed-length behaviour vector, and a short plain-English description.

Shared by build_cases.py (indexes past cases) and similar_cases.py (looks up neighbours).
16 dimensions, log-scaled where values span orders of magnitude, then z-scored with statistics from the TRAIN
window (saved to models/embed_meta.json) and clipped to [-5, 5], so no single column dominates cosine distance.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "models" / "embed_meta.json"
COLS = ["total_in", "max_in", "n_in", "n_out", "uniq_senders", "uniq_recipients", "passthrough_ratio", "max_fwd_frac",
        "min_dwell_s", "med_dwell_s", "n_fast_pass", "n_pass", "net_imbalance", "active_span_s", "max_txn_10m", "fan_ratio"]
DIM_NAMES = ["log_total_in", "log_max_in", "log_n_in", "log_n_out", "log_uniq_senders", "log_uniq_recipients",
             "passthrough_ratio", "max_fwd_frac", "log_min_dwell", "log_med_dwell", "log_n_fast_pass", "log_n_pass",
             "net_imbalance", "log_active_span", "log_max_txn_10m", "fan_ratio"]
DIM = len(DIM_NAMES)
NO_FORWARD_SECS = 6 * 3600     # "never forwarded within the horizon" is encoded as a long dwell, not as missing


def fetch_features(conn, split, accts=None):
    sql = f"SELECT acct_id, {', '.join(COLS)} FROM features_{split}"
    params = None
    if accts is not None:
        sql += " WHERE acct_id = ANY(%s)"
        params = (list(accts),)
    cur = conn.execute(sql, params)
    return pd.DataFrame.from_records(cur.fetchall(), columns=[d.name for d in cur.description], coerce_float=True)


def raw_matrix(df):
    f = df[COLS].astype(float)
    g = lambda c: f[c].fillna(0).clip(lower=0)
    return np.column_stack([
        np.log1p(g("total_in")), np.log1p(g("max_in")), np.log1p(g("n_in")), np.log1p(g("n_out")),
        np.log1p(g("uniq_senders")), np.log1p(g("uniq_recipients")),
        g("passthrough_ratio").clip(upper=2.0), g("max_fwd_frac").clip(upper=1.0),
        np.log1p(f["min_dwell_s"].fillna(NO_FORWARD_SECS).clip(lower=0)),
        np.log1p(f["med_dwell_s"].fillna(NO_FORWARD_SECS).clip(lower=0)),
        np.log1p(g("n_fast_pass")), np.log1p(g("n_pass")), g("net_imbalance").clip(upper=1.0),
        np.log1p(g("active_span_s")), np.log1p(g("max_txn_10m")), g("fan_ratio").clip(upper=10.0)])


def fit_scaler(raw):
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    return mean, np.where(std < 1e-6, 1.0, std)


def embed(raw, mean, std):
    return np.clip((raw - mean) / std, -5, 5)


def save_scaler(mean, std):
    META.parent.mkdir(exist_ok=True)
    META.write_text(json.dumps({"dims": DIM_NAMES, "mean": mean.tolist(), "std": std.tolist()}, indent=2))


def load_scaler():
    m = json.loads(META.read_text())
    return np.array(m["mean"]), np.array(m["std"])


def vec_text(v):
    return "[" + ",".join(f"{x:.5f}" for x in v) + "]"


def describe(r):
    """One-sentence behaviour summary from a feature row (dict-like with the COLS fields)."""
    def n(x):
        return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    dwell = r.get("min_dwell_s")
    fastest = "" if dwell is None or np.isnan(dwell) else f", fastest forward {dwell:.0f}s after receipt"
    return (f"received Rs {n(r['total_in']):,.0f} from {n(r['uniq_senders']):.0f} sender(s) and paid {n(r['uniq_recipients']):.0f} "
            f"recipient(s); up to {n(r['max_fwd_frac']):.0%} of a large inbound was forwarded onward{fastest}; "
            f"{n(r['n_fast_pass']):.0f} pass-through(s) within 2 minutes")
