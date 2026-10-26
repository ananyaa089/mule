"""Index past investigation cases in the pgvector table `cases`, then check retrieval quality.

    python src/build_cases.py

Cases = every TRAIN-window account that is a confirmed mule (injected or IBM pattern) plus the legitimate
look-alike decoys, each embedded from its behaviour (src/embedding.py). Only the train window is indexed, so when
a validation/test account is investigated its neighbours are all "past" cases.
Needs the pgvector extension (included in the pgvector/pgvector:pg16 Docker image).
"""
import os
from collections import Counter

import numpy as np
import psycopg

import embedding as emb

DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
VTYPE = os.environ.get("MULEGUARD_VECTOR_TYPE", f"vector({emb.DIM})")      # override only for testing without pgvector
K = 5


def case_rows(conn, split):
    cur = conn.execute(f"""
        SELECT l.acct_id, lb.case_id, CASE WHEN l.y = 1 THEN 'mule' ELSE 'decoy' END AS kind,
               l.typology, l.source
        FROM labels_{split} l LEFT JOIN labels lb USING (acct_id)
        WHERE l.y = 1 OR l.label = 'decoy'""")
    return cur.fetchall()


def main():
    with psycopg.connect(DSN, autocommit=True) as conn:
        try:
            if not os.environ.get("MULEGUARD_SKIP_HNSW"): conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.errors.Error as e:
            raise SystemExit(f"pgvector is not available in this Postgres ({e.__class__.__name__}). "
                             f"Use the pgvector/pgvector:pg16 image from docker-compose.yml.")

        train = emb.fetch_features(conn, "train")
        mean, std = emb.fit_scaler(emb.raw_matrix(train))              # scaler fitted on ALL train accounts
        emb.save_scaler(mean, std)

        meta = {r[0]: r for r in case_rows(conn, "train")}
        cases = train[train["acct_id"].isin(meta)].reset_index(drop=True)
        V = emb.embed(emb.raw_matrix(cases), mean, std)

        conn.execute("DROP TABLE IF EXISTS cases")
        conn.execute(f"""CREATE TABLE cases (acct_id TEXT PRIMARY KEY, case_id TEXT, kind TEXT NOT NULL,
                         typology TEXT, source TEXT, summary TEXT, embedding {VTYPE})""")
        with conn.cursor() as cur, cur.copy("COPY cases (acct_id, case_id, kind, typology, source, summary, embedding) FROM STDIN") as cp:
            for i, row in cases.iterrows():
                _, cid, kind, typ, src = meta[row["acct_id"]]
                cp.write_row((row["acct_id"], cid, kind, typ, src, emb.describe(row.to_dict()), emb.vec_text(V[i])))
        if not os.environ.get("MULEGUARD_SKIP_HNSW"):
            conn.execute("CREATE INDEX cases_hnsw ON cases USING hnsw (embedding vector_cosine_ops)")
        n_m = int((cases["acct_id"].map(lambda a: meta[a][2]) == "mule").sum())
        print(f"indexed {len(cases):,} past cases from the train window: {n_m:,} mules, {len(cases) - n_m:,} decoys")

        # ---- retrieval quality on the TEST window (neighbours come from train only)
        test = emb.fetch_features(conn, "test")
        tmeta = {r[0]: r for r in case_rows(conn, "test")}
        t = test[test["acct_id"].isin(tmeta)].reset_index(drop=True)
        T = emb.embed(emb.raw_matrix(t), mean, std)
        conf = Counter()
        typ_hit = typ_tot = 0
        for i, row in t.iterrows():
            _, _, kind, typ, src = tmeta[row["acct_id"]]
            nb = conn.execute(f"SELECT kind, typology FROM cases ORDER BY embedding <=> %s::{VTYPE.split('(')[0]} LIMIT {K}",
                              (emb.vec_text(T[i]),)).fetchall()
            pred = "mule" if sum(1 for k, _ in nb if k == "mule") * 2 > K else "decoy"
            conf[(kind, pred)] += 1
            if kind == "mule" and src == "injected":
                typ_tot += 1
                mt = Counter(ty for k, ty in nb if k == "mule").most_common(1)
                typ_hit += bool(mt) and mt[0][0] == typ

        print(f"\nretrieval check on test window ({len(t):,} accounts, majority of top-{K} neighbours):")
        for kind in ("mule", "decoy"):
            tot = conf[(kind, "mule")] + conf[(kind, "decoy")]
            print(f"  true {kind:<5}: {conf[(kind, kind)]:>5} / {tot:<5} retrieve a matching majority ({conf[(kind, kind)] / max(tot, 1):.1%})")
        print(f"  injected mules: neighbours' top typology equals the true typology for {typ_hit} / {typ_tot} ({typ_hit / max(typ_tot, 1):.1%})")
        print("  (IBM-pattern mules usually look like ordinary accounts, so low scores on them are expected.)")


if __name__ == "__main__":
    main()
