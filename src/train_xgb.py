"""Train and evaluate the supervised mule model (XGBoost) on the SQL (+ graph) feature tables.

    python src/train_xgb.py

Reads  features_*, labels_*  (db/build_features.py) and graph_features_*  (src/graph_features.py, optional)
Writes models/xgb_<name>.json, models/model_meta.json, table ml_scores (scores of the main model)

Models trained:
  tabular        - SQL features only (no graph)
  graph          - tabular + graph features            <- main model when graph tables exist
  graph_no_synth - graph, minus the INVENTED columns account_age_days / kyc_level / account_type
Comparing them shows what the graph adds and whether the model leans on invented columns.
Early stopping + decision threshold use the validation window; the test window is used once.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
SEED = 42
SYNTHETIC = ["account_age_days", "kyc_level", "account_type"]
LOG_COLS = ["total_in", "total_out", "avg_in", "avg_out", "max_in", "max_out", "active_span_s"]
KYC = {"full": 0, "min": 1, "none": 2}
LABEL_COLS = ["y", "label", "typology", "source"]


def load(conn, split: str) -> pd.DataFrame:
    has_g = conn.execute("SELECT to_regclass(%s)", (f"graph_features_{split}",)).fetchone()[0] is not None
    join = f"LEFT JOIN graph_features_{split} g USING (acct_id)" if has_g else ""
    q = f"""SELECT f.*, l.y, l.label, l.typology, l.source
            FROM features_{split} f JOIN labels_{split} l USING (acct_id)"""
    if has_g:
        gcols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s AND column_name <> 'acct_id' "
            "ORDER BY ordinal_position", (f"graph_features_{split}",)).fetchall()]
        q = f"""SELECT f.*, {', '.join('g.' + c for c in gcols)}, l.y, l.label, l.typology, l.source
                FROM features_{split} f JOIN labels_{split} l USING (acct_id) {join}"""
    with conn.cursor() as cur:
        cur.execute(q)
        names = [d.name for d in cur.description]
        return pd.DataFrame.from_records(cur.fetchall(), columns=names, coerce_float=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric model matrix. Amounts get log1p because they span 0 .. 1e12."""
    X = df.drop(columns=["acct_id"] + LABEL_COLS).copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    X["kyc_level"] = X["kyc_level"].map(KYC)
    X["account_type"] = (X["account_type"] == "business").astype(int)
    gcols = [c for c in X.columns if c.startswith("g_")]
    X[gcols] = X[gcols].fillna(0)
    return X.astype("float64")          # NaN in dwell-time columns is handled natively by xgboost


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[int(np.nanargmax(f1[:-1]))])


def metrics_at(y, s, thr):
    pred = s >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / max(p + r, 1e-12), 4))


def precision_at_k(y, s, k):
    return round(float(y[np.argsort(-s)[:k]].mean()), 4)


def train(Xtr, ytr, Xva, yva, label):
    spw = float(np.sqrt((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6, min_child_weight=3, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=spw, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"  [{label}] {Xtr.shape[1]} features, best round {model.best_iteration}, val PR-AUC {model.best_score:.4f}")
    return model


def typology_recall(test_df, yt, flagged):
    d = pd.DataFrame({"typology": test_df["typology"], "y": yt, "f": flagged}).query("y == 1")
    g = d.groupby("typology").agg(accounts=("y", "size"), caught=("f", "sum"))
    g["recall"] = (g["caught"] / g["accounts"]).round(3)
    return g


def main():
    MODELS.mkdir(exist_ok=True)
    with psycopg.connect(DSN) as conn:
        dfs = {s: load(conn, s) for s in ("train", "val", "test")}
    for s, d in dfs.items():
        print(f"{s:<5} {len(d):>8,} accounts  {int(d['y'].sum()):>6,} mules ({d['y'].mean():.3%})")

    X = {s: prepare(d) for s, d in dfs.items()}
    y = {s: d["y"].to_numpy() for s, d in dfs.items()}
    all_cols = list(X["train"].columns)
    has_graph = any(c.startswith("g_") for c in all_cols)
    sets = {"tabular": [c for c in all_cols if not c.startswith("g_")]}
    if has_graph:
        sets["graph"] = all_cols
        sets["graph_no_synth"] = [c for c in all_cols if c not in SYNTHETIC]
    else:
        print("(no graph_features_* tables found: training tabular only. Run src/graph_features.py first.)")
    main_name = "graph" if has_graph else "tabular"

    print("\ntraining")
    models = {n: train(X["train"][c], y["train"], X["val"][c], y["val"], n) for n, c in sets.items()}

    test_df, yt = dfs["test"], y["test"]
    print("\n=== TEST results (threshold chosen on validation) ===")
    results, scores, flags = {}, {}, {}
    for n, m in models.items():
        c = sets[n]
        sv, st = m.predict_proba(X["val"][c])[:, 1], m.predict_proba(X["test"][c])[:, 1]
        thr = best_f1_threshold(y["val"], sv)
        res = dict(threshold=thr, pr_auc=round(average_precision_score(yt, st), 4), roc_auc=round(roc_auc_score(yt, st), 4),
                   **metrics_at(yt, st, thr), precision_at_100=precision_at_k(yt, st, 100),
                   precision_at_500=precision_at_k(yt, st, 500), precision_at_1000=precision_at_k(yt, st, 1000))
        results[n], scores[n], flags[n] = res, st, st >= thr
        print(f"\n[{n}]  PR-AUC {res['pr_auc']}  ROC-AUC {res['roc_auc']}  threshold {thr:.3f}")
        print(f"   precision {res['precision']}  recall {res['recall']}  F1 {res['f1']}   (TP {res['tp']}, FP {res['fp']}, FN {res['fn']})")
        print(f"   precision@100 {res['precision_at_100']}   @500 {res['precision_at_500']}   @1000 {res['precision_at_1000']}")

    rule = (test_df["n_fast_pass"] >= 1).to_numpy()
    tp, fp = int((rule & (yt == 1)).sum()), int((rule & (yt == 0)).sum())
    print(f"\n[baseline rule: n_fast_pass >= 1]  precision {tp / max(tp + fp, 1):.4f}  recall {tp / max(int(yt.sum()), 1):.4f}")

    fl = flags[main_name]
    print(f"\n=== breakdown, main model '{main_name}' ===")
    rows = []
    for key, mask in [("injected mules", (yt == 1) & (test_df["source"] == "injected")),
                      ("IBM-pattern mules", (yt == 1) & (test_df["source"] == "ibm")),
                      ("decoys (legit look-alikes)", (test_df["label"] == "decoy").to_numpy()),
                      ("everyone else", ((yt == 0) & (test_df["label"] != "decoy")).to_numpy())]:
        rows.append((key, int(mask.sum()), int((fl & mask).sum()), round(float((fl & mask).sum() / max(mask.sum(), 1)), 4)))
    print(pd.DataFrame(rows, columns=["group", "accounts", "flagged", "share_flagged"]).to_string(index=False))

    print("\nrecall by mule typology (test)" + (":  tabular vs graph" if has_graph else ":"))
    tr = typology_recall(test_df, yt, flags["tabular"]).rename(columns={"caught": "caught_tab", "recall": "recall_tab"})
    if has_graph:
        tg = typology_recall(test_df, yt, fl).rename(columns={"caught": "caught_graph", "recall": "recall_graph"})
        tr = tr.join(tg[["caught_graph", "recall_graph"]])
    print(tr.sort_values("accounts", ascending=False).to_string())

    m, c = models[main_name], sets[main_name]
    sample = X["test"][c].sample(min(20000, len(X["test"])), random_state=SEED)
    contrib = m.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=c).sort_values(ascending=False)
    print(f"\ntop features by mean |contribution| ('{main_name}'):")
    print((imp / imp.sum()).round(3).head(12).to_string())

    if has_graph:
        d = results["graph"]["pr_auc"] - results["tabular"]["pr_auc"]
        g2 = results["graph"]["pr_auc"] - results["graph_no_synth"]["pr_auc"]
        ibm = (yt == 1) & (test_df["source"] == "ibm")
        r_t, r_g = (flags["tabular"] & ibm).sum() / max(ibm.sum(), 1), (flags["graph"] & ibm).sum() / max(ibm.sum(), 1)
        print(f"\nGraph effect: PR-AUC {results['tabular']['pr_auc']} -> {results['graph']['pr_auc']} ({d:+.4f}); "
              f"IBM-pattern mule recall {r_t:.1%} -> {r_g:.1%}")
        print(f"Invented columns: graph vs graph_no_synth PR-AUC gap {g2:+.4f} "
              f"({'model leans on them; report graph_no_synth' if g2 > 0.05 else 'not carrying the result'})")

    for n, mm in models.items():
        mm.save_model(MODELS / f"xgb_{n}.json")
    (MODELS / "model_meta.json").write_text(json.dumps(
        {"feature_sets": sets, "log_cols": LOG_COLS, "kyc_map": KYC, "results": results, "main_model": main_name}, indent=2))
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS ml_scores")
        conn.execute("CREATE TABLE ml_scores (split TEXT, acct_id TEXT, score DOUBLE PRECISION, PRIMARY KEY (split, acct_id))")
        with conn.cursor() as cur, cur.copy("COPY ml_scores (split, acct_id, score) FROM STDIN") as cp:
            for s in ("train", "val", "test"):
                sc = models[main_name].predict_proba(X[s][sets[main_name]])[:, 1]
                for a, v in zip(dfs[s]["acct_id"], sc):
                    cp.write_row((s, a, float(v)))
    print("\nsaved models/ and table ml_scores.")


if __name__ == "__main__":
    main()
