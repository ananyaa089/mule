"""Step 4 - build the accounts table and the merged labels table.

Output: data/final/accounts.parquet
        data/final/labels.parquet
        data/final/patterns.parquet

The datasets have no account metadata, so opened_at / kyc_level / account_type are
SYNTHETIC enrichment. Injected mules get a mild bias (younger, weaker KYC) so the
features carry some signal without being a giveaway. Say so in your report.
"""
import sys

import numpy as np
import pandas as pd

from config import CLEAN, FINAL, SEED

BUSINESS_DECOYS = {"biz_supplier", "aggregator", "trusted_forwarder"}


def main() -> None:
    tx_file = FINAL / "transactions.parquet"
    if not tx_file.exists():
        sys.exit("Run 03_inject.py first.")
    rng = np.random.default_rng(SEED + 2)

    tx = pd.read_parquet(tx_file, columns=["ts", "src", "dst"])
    first = pd.concat([tx.groupby("src")["ts"].min(), tx.groupby("dst")["ts"].min()]).groupby(level=0).min()
    counts = pd.concat([tx["src"], tx["dst"]], ignore_index=True).value_counts()
    del tx

    labels = pd.concat([pd.read_parquet(CLEAN / "labels_base.parquet"),
                        pd.read_parquet(CLEAN / "labels_injected.parquet")], ignore_index=True)
    assert labels["acct_id"].is_unique, "an account has two labels - check injection pools"
    labels["is_mule"] = labels["label"] == "mule"

    acc = pd.DataFrame({"acct_id": first.index.to_numpy(), "first_ts": first.to_numpy()})
    acc["n_txn"] = counts.reindex(acc["acct_id"]).to_numpy()
    lab = labels.set_index("acct_id")
    acc["label"] = acc["acct_id"].map(lab["label"])
    acc["role"] = acc["acct_id"].map(lab["role"])
    acc["source"] = acc["acct_id"].map(lab["source"])
    n = len(acc)

    inj_mule = ((acc["label"] == "mule") & (acc["source"] == "injected")).to_numpy()
    decoy = (acc["label"] == "decoy").to_numpy()
    biz_decoy = (decoy & acc["role"].isin(BUSINESS_DECOYS)).to_numpy()

    # ---- opened_at: days before first activity (log-uniform 1 month .. 10 years)
    days = np.exp(rng.uniform(np.log(30), np.log(3650), n))
    young = inj_mule & (rng.random(n) < 0.60)
    days[young] = rng.uniform(1, 45, young.sum())
    days[biz_decoy] = rng.uniform(180, 3650, biz_decoy.sum())
    opened = acc["first_ts"].dt.normalize() - pd.to_timedelta(np.ceil(days), unit="D")

    # ---- kyc level
    def draw(p_full, p_min):
        u = rng.random(n)
        return np.where(u < p_full, "full", np.where(u < p_full + p_min, "min", "none"))
    kyc = draw(0.85, 0.12)
    kyc = np.where(inj_mule, draw(0.55, 0.33), kyc)
    kyc = np.where(decoy, draw(0.90, 0.10), kyc)

    # ---- account type: busiest ~3% of ordinary accounts are business
    ordinary = ~(inj_mule | decoy)
    cut = np.quantile(acc.loc[ordinary, "n_txn"], 0.97)
    atype = np.where(ordinary & (acc["n_txn"].to_numpy() >= cut), "business", "personal")
    atype = np.where(inj_mule & (rng.random(n) < 0.15), "business", atype)
    atype = np.where(biz_decoy, "business", atype)

    accounts = pd.DataFrame({
        "acct_id": acc["acct_id"],
        "bank": acc["acct_id"].str.partition("_")[0],
        "opened_at": opened,
        "kyc_level": kyc,
        "account_type": atype,
        "status": "active",
    })

    accounts.to_parquet(FINAL / "accounts.parquet", index=False)
    labels.to_parquet(FINAL / "labels.parquet", index=False)
    pd.read_parquet(CLEAN / "patterns.parquet").to_parquet(FINAL / "patterns.parquet", index=False)

    print(f"accounts : {len(accounts):,}")
    print(f"labelled : {len(labels):,}  (mules {int(labels['is_mule'].sum()):,}, "
          f"decoys {int((labels['label'] == 'decoy').sum()):,})")
    print(f"mule share of accounts: {labels['is_mule'].sum() / len(accounts):.3%}")
    print("kyc mix:", accounts["kyc_level"].value_counts(normalize=True).round(3).to_dict())
    print("type mix:", accounts["account_type"].value_counts(normalize=True).round(3).to_dict())
    print(f"wrote accounts.parquet, labels.parquet, patterns.parquet to {FINAL}")


if __name__ == "__main__":
    main()
