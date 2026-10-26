"""Graph features per time window -> tables graph_features_{train,val,test}.

    python src/graph_features.py

Why: dwell-time features cannot see slow, structural laundering (cycles, scatter-gather, bipartite, stack).
These features describe each account's position in the transaction graph of ITS OWN WINDOW only
(same windows as db/build_features.py, read from the `splits` table). No labels are used.

Hub accounts (many counterparties) are excluded from the structural counts: they sit on huge numbers of
paths and would drown out the small structures that laundering patterns form. They still count for the
neighbour averages.
Time order of edges is ignored here (prototype simplification; the fund-trace step is time-aware).
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy import sparse
from scipy.sparse.csgraph import connected_components

DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
LOWDEG = int(os.environ.get("MULEGUARD_LOWDEG", 25))   # nodes with more distinct counterparties are 'hubs'


def rowsum(M):
    return np.asarray(M.sum(axis=1)).ravel()


def colsum(M):
    return np.asarray(M.sum(axis=0)).ravel()


def compute(n, src, dst, f):
    """src/dst: int arrays of distinct directed edges. f: DataFrame of tabular features, row i = node i."""
    A = sparse.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n))
    A.data[:] = 1.0
    AT = A.T.tocsr()
    indeg, outdeg = colsum(A), rowsum(A)
    out = {"g_recip": rowsum(A.multiply(AT))}                      # counterparties that pay me AND I pay

    low = ((indeg + outdeg) <= LOWDEG).astype(np.float32)
    D = sparse.diags(low)
    AL = (D @ A @ D).tocsr()
    AL.eliminate_zeros()
    ALT = AL.T.tocsr()
    out["g_low_in"], out["g_low_out"] = colsum(AL), rowsum(AL)

    P = (AL @ AL).tocsr()                                           # P[a,c] = number of 2-step paths a->b->c
    P.setdiag(0)
    P.eliminate_zeros()
    out["g_cycle3"] = rowsum(P.multiply(ALT))                       # closes back to a: 3-cycles through a
    out["g_reach2_out"] = P.getnnz(axis=1).astype(float)
    out["g_reach2_in"] = P.getnnz(axis=0).astype(float)

    Q = P.copy()                                                    # pairs (s,t) joined by >= 2 different middles
    Q.data = (Q.data >= 2).astype(np.float32)
    Q.eliminate_zeros()
    out["g_diamond_src"] = Q.getnnz(axis=1).astype(float)           # scatter-gather source
    out["g_diamond_sink"] = Q.getnnz(axis=0).astype(float)          # scatter-gather sink
    out["g_diamond_mid"] = rowsum((ALT @ Q).multiply(AL))           # lies between such a pair

    for key, M in (("g_co_send", AL @ ALT), ("g_co_recv", ALT @ AL)):   # accounts sharing >= 2 recipients / senders
        S = M.tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        S.data = (S.data >= 2).astype(np.float32)
        S.eliminate_zeros()
        out[key] = S.getnnz(axis=1).astype(float)

    out["g_scc_size"] = np.bincount(l := connected_components(AL, True, "strong")[1])[l].astype(float)
    out["g_wcc_size"] = np.bincount(l := connected_components(AL, False)[1])[l].astype(float)

    # neighbour averages ("what do my counterparties look like"), on the full graph
    interm = ((f["n_in"].to_numpy() > 0) & (f["n_out"].to_numpy() > 0) & (low > 0)).astype(np.float32)
    vecs = {"loginflow": np.log1p(f["total_in"].to_numpy()).astype(np.float32),
            "uniq_s": np.log1p(f["uniq_senders"].to_numpy()).astype(np.float32),
            "uniq_r": np.log1p(f["uniq_recipients"].to_numpy()).astype(np.float32),
            "fastpass": (f["n_fast_pass"].to_numpy() > 0).astype(np.float32),
            "interm": interm, "low": low}
    do, di = np.maximum(outdeg, 1), np.maximum(indeg, 1)
    for k, v in vecs.items():
        out[f"g_nbout_{k}"] = (A @ v) / do
        out[f"g_nbin_{k}"] = (AT @ v) / di
    return pd.DataFrame(out).astype("float64")


def edges_for_window(conn, name, start, end):
    """Return (acct_ids, features df, src idx, dst idx) with node index = row number in features_{name} order."""
    cur = conn.execute(f"SELECT acct_id, n_in, n_out, uniq_senders, uniq_recipients, total_in, n_fast_pass "
                       f"FROM features_{name} ORDER BY acct_id")
    cols = [d.name for d in cur.description]
    f = pd.DataFrame.from_records(cur.fetchall(), columns=cols, coerce_float=True)
    chunks = []
    with conn.transaction():                                       # named cursors need a transaction
        cur = conn.cursor(name=f"edges_{name}")                    # server-side: no giant fetchall
        cur.execute(f"""
            WITH n AS (SELECT acct_id, (row_number() OVER (ORDER BY acct_id) - 1)::int AS idx FROM features_{name})
            SELECT s.idx, d.idx
            FROM (SELECT DISTINCT src, dst FROM transactions WHERE ts >= %s AND ts < %s AND src <> dst) e
            JOIN n s ON s.acct_id = e.src JOIN n d ON d.acct_id = e.dst""", (start, end))
        while rows := cur.fetchmany(500_000):
            chunks.append(np.asarray(rows, dtype=np.int32))
        cur.close()
    e = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.int32)
    return f, e[:, 0], e[:, 1]


def write_table(conn, name, acct_ids, G):
    t = f"graph_features_{name}"
    cols = list(G.columns)
    conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.execute(f"CREATE TABLE {t} (acct_id TEXT PRIMARY KEY, " + ", ".join(f"{c} DOUBLE PRECISION" for c in cols) + ")")
    with conn.cursor() as cur, cur.copy(f"COPY {t} (acct_id, {', '.join(cols)}) FROM STDIN WITH (FORMAT csv)") as cp:
        for i in range(0, len(G), 200_000):
            part = G.iloc[i:i + 200_000].copy()
            part.insert(0, "acct_id", acct_ids[i:i + 200_000])
            cp.write(part.to_csv(header=False, index=False))
    conn.execute(f"ANALYZE {t}")


def main():
    with psycopg.connect(DSN, autocommit=True) as conn:
        splits = conn.execute("SELECT name, start_ts, end_ts FROM splits ORDER BY start_ts").fetchall()
        for name, start, end in splits:
            t = time.time()
            f, src, dst = edges_for_window(conn, name, start, end)
            G = compute(len(f), src, dst, f)
            write_table(conn, name, f["acct_id"].tolist(), G)
            print(f"[{name}] {len(f):,} nodes, {len(src):,} edges -> graph_features_{name} "
                  f"({G.shape[1]} cols, {time.time() - t:.0f}s)", flush=True)

        print("\nSanity check, TRAIN window - share of accounts with each structure, by group:")
        q = """SELECT CASE WHEN l.y = 1 AND l.source = 'ibm' THEN 'IBM-pattern mule'
                        WHEN l.y = 1 THEN 'injected mule'
                        WHEN l.label = 'decoy' THEN 'decoy' ELSE 'everyone else' END AS grp,
                      count(*) accounts,
                      round(avg((g.g_cycle3 > 0)::int)::numeric, 3)      in_3cycle,
                      round(avg((g.g_scc_size > 1)::int)::numeric, 3)    in_cycle_scc,
                      round(avg((g.g_diamond_mid > 0)::int)::numeric, 3) diamond_mid,
                      round(avg((g.g_co_send > 0)::int)::numeric, 3)     co_send,
                      round(avg((g.g_co_recv > 0)::int)::numeric, 3)     co_recv,
                      round(avg((g.g_recip > 0)::int)::numeric, 3)       reciprocal
               FROM graph_features_train g JOIN labels_train l USING (acct_id) GROUP BY 1 ORDER BY 1"""
        cur = conn.execute(q)
        print(pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description]).to_string(index=False))
    print("\nDone. Next: python src/train_xgb.py (it now compares tabular vs tabular+graph).")


if __name__ == "__main__":
    main()
