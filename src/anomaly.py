import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
SEED = 42
NO_FORWARD_SECS = 6 * 3600
COLS = ["n_in", "n_out", "uniq_senders", "uniq_recipients", "total_in", "total_out", "avg_in", "avg_out", "max_in",
        "max_out", "passthrough_ratio", "sender_diversity", "recipient_diversity", "fan_ratio", "net_imbalance",
        "active_span_s", "max_txn_10m", "n_large_in", "n_pass", "n_fast_pass", "min_dwell_s", "med_dwell_s", "max_fwd_frac"]
LOG = {"n_in", "n_out", "uniq_senders", "uniq_recipients", "total_in", "total_out", "avg_in", "avg_out", "max_in", "max_out",
       "active_span_s", "max_txn_10m", "n_large_in", "n_pass", "n_fast_pass", "min_dwell_s", "med_dwell_s"}
CAP = {"passthrough_ratio": 10.0, "fan_ratio": 10.0, "net_imbalance": 1.0, "max_fwd_frac": 1.0,
       "sender_diversity": 1.0, "recipient_diversity": 1.0}
# deliberately NOT used: account_age_days, kyc_level, account_type (invented columns)


def matrix(df):
    out = {}
    for c in COLS:
        v = df[c].astype(float)
        v = v.fillna(NO_FORWARD_SECS if c in ("min_dwell_s", "med_dwell_s") else 0.0).clip(lower=0)
        if c in CAP:
            v = v.clip(upper=CAP[c])
        out[c] = np.log1p(v) if c in LOG else v
    return pd.DataFrame(out)[COLS].to_numpy(dtype=np.float32)


class AE(nn.Module):
    def __init__(self, d, h=16, z=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def errors(model, X, per_feature=False, bs=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs])
            e = (model(xb) - xb) ** 2
            out.append(e.numpy() if per_feature else e.mean(1).numpy())
    return np.concatenate(out)


def fetch(conn, split, accts=None):
    sql = f"SELECT acct_id, {', '.join(COLS)} FROM features_{split}"
    params = None
    if accts is not None:
        sql += " WHERE acct_id = ANY(%s)"
        params = (list(accts),)
    cur = conn.execute(sql, params)
    return pd.DataFrame.from_records(cur.fetchall(), columns=[d.name for d in cur.description], coerce_float=True)


def fetch_labels(conn, split):
    cur = conn.execute(f"SELECT acct_id, y, label, source, typology FROM labels_{split}")
    return pd.DataFrame.from_records(cur.fetchall(), columns=["acct_id", "y", "label", "source", "typology"])


def train_model(X, epochs=20, bs=2048):
    torch.manual_seed(SEED)
    model = AE(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(SEED + ep).permutation(n)
        total = 0.0
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[perm[i:i + bs]])
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:>2}/{epochs}  train loss {total / n:.5f}", flush=True)
    return model


def main():
    MODELS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    with psycopg.connect(DSN, autocommit=True) as conn:
        feats = {s: fetch(conn, s) for s in ("train", "val", "test")}
        lab = {s: fetch_labels(conn, s) for s in feats}

    tr = feats["train"].merge(lab["train"], on="acct_id")
    normal = tr[tr["y"] == 0]                                   # everything that is not a confirmed mule (decoys included)
    Xn = matrix(normal)
    mean, std = Xn.mean(0), Xn.std(0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    Z = lambda df: np.clip((matrix(df) - mean) / std, -8, 8).astype(np.float32)
    print(f"training on {len(normal):,} normal train-window accounts, {len(COLS)} features")
    model = train_model(Z(normal))

    ref = np.sort(errors(model, Z(normal)))                      # error distribution of normal accounts -> percentile scale
    torch.save(model.state_dict(), MODELS / "anomaly_ae.pt")
    (MODELS / "anomaly_meta.json").write_text(json.dumps(
        {"cols": COLS, "mean": mean.tolist(), "std": std.tolist(), "ref_errors_quantiles": np.quantile(ref, np.linspace(0, 1, 1001)).tolist()}))

    res = {}
    for s, f in feats.items():
        e = errors(model, Z(f))
        res[s] = pd.DataFrame({"acct_id": f["acct_id"], "recon_error": e, "anomaly_pct": np.searchsorted(ref, e) / len(ref)})
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS anomaly_scores")
        conn.execute("CREATE TABLE anomaly_scores (split TEXT, acct_id TEXT, recon_error DOUBLE PRECISION, anomaly_pct DOUBLE PRECISION, PRIMARY KEY (split, acct_id))")
        with conn.cursor() as cur, cur.copy("COPY anomaly_scores (split, acct_id, recon_error, anomaly_pct) FROM STDIN") as cp:
            for s, d in res.items():
                for a, e, p in zip(d["acct_id"], d["recon_error"], d["anomaly_pct"]):
                    cp.write_row((s, a, float(e), float(p)))
        try:
            ml = pd.DataFrame(conn.execute("SELECT acct_id, score FROM ml_scores WHERE split = 'test'").fetchall(), columns=["acct_id", "ml"])
            thr = json.loads((MODELS / "model_meta.json").read_text())
            thr = thr["results"][thr["main_model"]]["threshold"]
        except Exception:
            ml, thr = None, None

    te = res["test"].merge(lab["test"], on="acct_id")
    y, s = te["y"].to_numpy(), te["anomaly_pct"].to_numpy()
    print(f"\n=== TEST window ({len(te):,} accounts, {int(y.sum()):,} mules) - anomaly score used ON ITS OWN ===")
    print(f"ROC-AUC {roc_auc_score(y, s):.4f}   PR-AUC {average_precision_score(y, s):.4f}   (random PR-AUC would be {y.mean():.4f})")
    groups = {"injected mules": (y == 1) & (te["source"] == "injected"), "IBM-pattern mules": (y == 1) & (te["source"] == "ibm"),
              "decoys": (te["label"] == "decoy").to_numpy(), "everyone else": (y == 0) & (te["label"] != "decoy")}
    rows = []
    for g, m in groups.items():
        rows.append((g, int(m.sum()), f"{(s[m] >= 0.99).mean():.1%}", f"{(s[m] >= 0.95).mean():.1%}", f"{np.median(s[m]):.3f}"))
    print(pd.DataFrame(rows, columns=["group", "accounts", "in top 1%", "in top 5%", "median anomaly_pct"]).to_string(index=False))
    if ml is not None:
        te = te.merge(ml, on="acct_id", how="left")
        missed = (te["y"] == 1) & (te["ml"].fillna(0) < thr)
        print(f"\nWhat the anomaly model adds beyond the supervised model (threshold {thr:.3f}):")
        print(f"  mules the supervised model MISSED: {int(missed.sum()):,}; of those, in anomaly top 5%: {int(((te['anomaly_pct'] >= 0.95) & missed).sum()):,} "
              f"({((te['anomaly_pct'] >= 0.95) & missed).sum() / max(missed.sum(), 1):.1%}), in top 1%: {int(((te['anomaly_pct'] >= 0.99) & missed).sum()):,}")
        flagged_normal = ((te["anomaly_pct"] >= 0.99) & (te["y"] == 0) & (te["label"] != "decoy")).sum()
        print(f"  ordinary accounts in the anomaly top 1%: {int(flagged_normal):,} (this is the review burden if you used it alone)")
    print("\nsaved models/anomaly_ae.pt and table anomaly_scores.")


def score_account(conn, acct):
    """Compact dict for the investigation report: anomaly percentile + the features that deviate most."""
    meta = json.loads((MODELS / "anomaly_meta.json").read_text())
    mean, std = np.array(meta["mean"], dtype=np.float32), np.array(meta["std"], dtype=np.float32)
    for split in ("test", "val", "train"):
        df = fetch(conn, split, [acct])
        if len(df):
            break
    else:
        return {"account": acct, "error": "account has no activity in any analysed window"}
    model = AE(len(meta["cols"]))
    model.load_state_dict(torch.load(MODELS / "anomaly_ae.pt", weights_only=True))
    z = np.clip((matrix(df) - mean) / std, -8, 8).astype(np.float32)
    pf = errors(model, z, per_feature=True)[0]
    q = np.array(meta["ref_errors_quantiles"])
    pct = float(np.searchsorted(q, pf.mean()) / len(q))
    raw = df.iloc[0]
    top = np.argsort(-pf)[:4]
    return {"account": acct, "window": split, "anomaly_pct": round(min(pct, 0.999), 3),
            "top_deviations": [{"feature": COLS[i], "value": None if pd.isna(raw[COLS[i]]) else round(float(raw[COLS[i]]), 2),
                                "z_vs_normal": round(float(z[0, i]), 1)} for i in top]}


if __name__ == "__main__":
    main()
