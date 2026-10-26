"""Build windowed feature tables: features_train / features_val / features_test
(+ matching labels_train / labels_val / labels_test).

    python db/build_features.py

Split by time AND traffic: first 70% of transactions (in time order) = train, next 15% = val, last 15% = test.
Settings via env: MULEGUARD_LARGE (default 25000), MULEGUARD_HORIZON_MIN (default 15).
"""
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
LARGE = float(os.environ.get("MULEGUARD_LARGE", 25_000))
HORIZON = int(os.environ.get("MULEGUARD_HORIZON_MIN", 15))
SPLITS = [("train", 0.0, 0.70), ("val", 0.70, 0.85), ("test", 0.85, 1.0)]


def main():
    template = (ROOT / "sql" / "04_features.sql").read_text()
    with psycopg.connect(DSN, autocommit=True) as conn:
        t_min, t_max = conn.execute("SELECT min(ts), max(ts) FROM transactions").fetchone()
        # Split by TRAFFIC (transaction-count quantiles), not calendar time: IBM traffic is dense early
        # and thin late, so equal calendar slices would leave val/test nearly empty.
        qs = {q: conn.execute("SELECT percentile_disc(%s) WITHIN GROUP (ORDER BY ts) FROM transactions",
                              (q,)).fetchone()[0] for q in (0.70, 0.85)}
        edges = {0.0: t_min, 0.70: qs[0.70], 0.85: qs[0.85], 1.0: t_max + timedelta(milliseconds=1)}
        conn.execute("DROP TABLE IF EXISTS splits")
        conn.execute("CREATE TABLE splits (name TEXT PRIMARY KEY, start_ts TIMESTAMP, end_ts TIMESTAMP)")
        for name, a, b in SPLITS:
            start, end = edges[a], edges[b]
            assert re.fullmatch(r"[a-z_]+", name)
            conn.execute("INSERT INTO splits VALUES (%s,%s,%s)", (name, start, end))
            print(f"[{name}] {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M}", flush=True)
            t = time.time()
            # client-side binding: CREATE TABLE AS cannot take server-side parameters
            cur = psycopg.ClientCursor(conn)
            cur.execute(template.replace("{name}", name),
                        {"s": start, "e": end, "large": LARGE, "horizon": HORIZON})
            n, pos = conn.execute(f"SELECT count(*), sum(y) FROM labels_{name}").fetchone()
            print(f"   features_{name}: {n:,} accounts, {pos:,} positive ({time.time() - t:.0f}s)", flush=True)

        print("\nSanity check - average feature values, mule vs decoy vs everything else, TRAIN window:")
        rows = conn.execute("""
            SELECT CASE WHEN l.y = 1 THEN 'mule' WHEN l.label = 'decoy' THEN 'decoy' ELSE 'other' END grp, count(*) n,
                   round(avg(f.n_fast_pass)::numeric, 3)        fast_pass,
                   round(avg(f.max_fwd_frac)::numeric, 3)       max_fwd,
                   round(avg(f.passthrough_ratio)::numeric, 3)  passthru,
                   round((percentile_cont(0.5) WITHIN GROUP (ORDER BY f.min_dwell_s))::numeric, 1) med_min_dwell_s,
                   round(avg(f.account_age_days)::numeric, 0)   age_days
            FROM features_train f JOIN labels_train l USING (acct_id) GROUP BY 1 ORDER BY 1
        """).fetchall()
        print("  group | accounts | avg n_fast_pass | avg max_fwd_frac | avg passthrough | median min_dwell_s | avg age_days")
        for r in rows:
            print("  " + " | ".join(str(x) for x in r))
    print("\nDone. Next: train the model on features_train/labels_train.")


if __name__ == "__main__":
    main()
