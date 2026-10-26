"""Bulk-load data/final/*.parquet into Postgres with COPY.

    python db/load_postgres.py            # schema + load + indexes
    python db/load_postgres.py --no-vector

Connection: env MULEGUARD_DSN, default postgresql://muleguard:muleguard@localhost:5433/muleguard
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
FINAL = Path(os.environ.get("MULEGUARD_DATA", ROOT / "data")) / "final"
SQL = ROOT / "sql"
DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")

# table -> (parquet file, columns in table order)
TABLES = {
    "accounts": ("accounts.parquet", ["acct_id", "bank", "opened_at", "kyc_level", "account_type", "status"]),
    "transactions": ("transactions.parquet", ["txn_id", "ts", "src", "dst", "amount", "currency", "channel",
                                              "channel_orig", "is_laundering", "is_injected", "case_id"]),
    "labels": ("labels.parquet", ["acct_id", "label", "role", "typology", "case_id", "source", "is_mule"]),
    "scenarios": ("scenarios.parquet", ["case_id", "kind", "typology", "primary_acct", "t0", "amount",
                                        "n_txns", "dwell_profile"]),
    "patterns": ("patterns.parquet", ["case_id", "typology", "n_txns", "n_accounts", "start_ts", "end_ts",
                                      "total_amount"]),
}


def copy_table(conn, table, fname, cols, batch_rows=250_000):
    path = FINAL / fname
    if not path.exists():
        sys.exit(f"Missing {path}. Run data_prep/run_all.py first.")
    pf = pq.ParquetFile(path)
    total = 0
    t = time.time()
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({','.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')") as cp:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            df = batch.to_pandas()
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                elif df[c].dtype == bool:
                    df[c] = df[c].map({True: "t", False: "f"})
            buf = io.StringIO()
            df.to_csv(buf, header=False, index=False)
            cp.write(buf.getvalue())
            total += len(df)
    print(f"  {table:<13} {total:>10,} rows  ({time.time() - t:.0f}s)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vector", action="store_true", help="skip the pgvector extension/table")
    args = ap.parse_args()

    print(f"connecting to {DSN.split('@')[-1]} ...")
    with psycopg.connect(DSN, autocommit=True) as conn:
        print("creating schema")
        conn.execute((SQL / "01_schema.sql").read_text())
    with psycopg.connect(DSN) as conn:
        print("loading")
        counts = {t: copy_table(conn, t, f, c) for t, (f, c) in TABLES.items()}
        conn.commit()
    with psycopg.connect(DSN, autocommit=True) as conn:
        print("building indexes + foreign keys (a minute or two)")
        t = time.time()
        conn.execute((SQL / "02_indexes.sql").read_text())
        print(f"  done ({time.time() - t:.0f}s)")
        if not args.no_vector:
            try:
                conn.execute((SQL / "03_vector.sql").read_text())
                print("pgvector table 'cases' ready")
            except psycopg.errors.Error as e:
                print(f"  pgvector unavailable ({e.__class__.__name__}); rerun later or use --no-vector")
        for t_, n in counts.items():
            got = conn.execute(f"SELECT count(*) FROM {t_}").fetchone()[0]
            assert got == n, f"{t_}: loaded {got} but expected {n}"
    print("all row counts match. Database ready.")


if __name__ == "__main__":
    main()
