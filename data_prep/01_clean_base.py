"""Step 1 - clean the raw IBM AMLworld HI-Small transaction file.

Input : data/raw/HI-Small_Trans.csv
Output: data/clean/transactions_base.parquet

What it does
  * reads bank/account codes as strings (leading zeros matter: "011" != "11")
  * keeps only same-currency rows and drops self-transfers ("Reinvestment")
  * builds one account id per account: "<bank>_<account>"
  * treats amounts as INR (relabel only, no rescaling)
  * adds random seconds + milliseconds to the minute-level timestamps
  * maps IBM payment formats to Indian channels (NEFT/RTGS/IMPS/UPI/...)
  * sorts by time
"""
import sys

import numpy as np
import pandas as pd

from config import CLEAN, RAW_TRANS, SEED

COLS = ["ts", "src_bank", "src_acct", "dst_bank", "dst_acct", "amt_recv", "cur_recv",
        "amount", "cur_paid", "channel_orig", "is_laundering"]


def remap_channels(orig: np.ndarray, amount: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IBM formats -> Indian channels.

    ACH/Wire are spread over several channels on purpose. If every base row were
    NEFT/RTGS while injected rows were IMPS/UPI, the channel column alone would
    reveal which rows are injected (label leakage).
    """
    n = len(orig)
    u = rng.random(n)
    out = np.full(n, "OTHER", dtype=object)

    ach = orig == "ACH"
    out[ach & (u < 0.40)] = "NEFT"
    out[ach & (u >= 0.40) & (u < 0.70)] = "IMPS"
    upi_slot = ach & (u >= 0.70)
    out[upi_slot & (amount <= 100_000)] = "UPI"
    out[upi_slot & (amount > 100_000)] = "NEFT"

    wire = orig == "Wire"
    big = amount >= 200_000
    out[wire & big & (u < 0.70)] = "RTGS"
    out[wire & big & (u >= 0.70)] = "NEFT"
    out[wire & ~big & (u < 0.50)] = "IMPS"
    out[wire & ~big & (u >= 0.50)] = "NEFT"

    out[orig == "Credit Card"] = "CARD"
    out[orig == "Cheque"] = "CHEQUE"
    out[orig == "Cash"] = "CASH"

    out[(out == "IMPS") & (amount > 500_000)] = "NEFT"   # IMPS cap is Rs 5 lakh
    return out


def main() -> None:
    if not RAW_TRANS.exists():
        sys.exit(f"Missing {RAW_TRANS}. Put HI-Small_Trans.csv in data/raw/ first.")
    CLEAN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    str_cols = {"src_bank": str, "src_acct": str, "dst_bank": str, "dst_acct": str}
    df = pd.read_csv(RAW_TRANS, header=0, names=COLS, dtype=str_cols)
    n_raw = len(df)
    print(f"raw rows                      : {n_raw:,}")

    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M", errors="coerce")
    df = df[df["ts"].notna()]

    df = df[df["cur_recv"] == df["cur_paid"]]
    print(f"after same-currency filter    : {len(df):,}")

    self_tr = (df["src_bank"] == df["dst_bank"]) & (df["src_acct"] == df["dst_acct"])
    print(f"self-transfers dropped        : {int(self_tr.sum()):,}")
    df = df[~self_tr]

    df = df[df["amount"] > 0].copy()
    print(f"clean rows                    : {len(df):,}")

    df["src"] = df["src_bank"] + "_" + df["src_acct"]
    df["dst"] = df["dst_bank"] + "_" + df["dst_acct"]

    # minute-level -> second/millisecond-level (random, reproducible)
    ms = rng.integers(0, 60_000, len(df))
    df["ts"] = df["ts"] + pd.to_timedelta(ms, unit="ms")

    amount = df["amount"].to_numpy()
    df["channel"] = remap_channels(df["channel_orig"].to_numpy(), amount, rng)
    df["is_laundering"] = df["is_laundering"].astype("int8")

    df = df[["ts", "src", "dst", "amount", "channel", "channel_orig", "is_laundering"]]
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

    print("\namount distribution (treated as INR):")
    print(df["amount"].describe(percentiles=[.5, .9, .99, .999]).round(2).to_string())
    print(f"share of txns >= 3,00,000     : {(df['amount'] >= 300_000).mean():.4%}")
    print(f"time range                    : {df['ts'].min()}  ->  {df['ts'].max()}")
    print(f"laundering rate               : {df['is_laundering'].mean():.4%}")
    print("\nchannel mix:")
    print(df["channel"].value_counts(normalize=True).round(3).to_string())

    out = CLEAN / "transactions_base.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
