"""Step 5 - sanity checks on the final dataset. Exits non-zero if anything fails."""
import sys

import pandas as pd

from config import (FAST_DWELL_S, FINAL, HOP2_DELAY_S, MEDIUM_DWELL_S)

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> None:
    tx = pd.read_parquet(FINAL / "transactions.parquet")
    acc = pd.read_parquet(FINAL / "accounts.parquet")
    lab = pd.read_parquet(FINAL / "labels.parquet")
    sc = pd.read_parquet(FINAL / "scenarios.parquet")
    inj = tx[tx["is_injected"]]
    by_case = {k: g for k, g in inj.groupby("case_id")}

    # ---------------------------------------------------------------- general
    print("== general ==")
    check("txn_id unique and time-ordered", tx["txn_id"].is_unique and tx["ts"].is_monotonic_increasing)
    check("amounts positive", bool((tx["amount"] > 0).all()))
    check("no self-transfers", bool((tx["src"] != tx["dst"]).all()))
    check("no missing timestamps", bool(tx["ts"].notna().all()))
    ids = set(acc["acct_id"])
    check("every txn account is in accounts", set(tx["src"]) <= ids and set(tx["dst"]) <= ids)
    check("every label account is in accounts", set(lab["acct_id"]) <= ids)
    check("accounts unique", acc["acct_id"].is_unique)
    first = pd.concat([tx.groupby("src")["ts"].min(), tx.groupby("dst")["ts"].min()]).groupby(level=0).min()
    opened = acc.set_index("acct_id")["opened_at"]
    check("opened_at before first transaction", bool((opened.reindex(first.index) < first).all()))

    # ---------------------------------------------------------------- mules
    print("\n== injected mules ==")
    bad_dwell = bad_ratio = bad_order = bad_l2 = 0
    for r in sc[sc["kind"] == "mule"].itertuples():
        g = by_case[r.case_id]
        inb, outs = g[g["dst"] == r.primary_acct], g[g["src"] == r.primary_acct]
        dwell = (outs["ts"].min() - inb["ts"].min()).total_seconds()
        ratio = outs["amount"].sum() / inb["amount"].sum()
        lo, hi = FAST_DWELL_S if r.dwell_profile == "fast" else MEDIUM_DWELL_S
        bad_order += dwell <= 0
        bad_dwell += not (lo - 0.01 <= dwell <= hi + 0.01)
        bad_ratio += not (0.899 <= ratio <= 0.991)
        if r.typology == "two_hop":
            for l2 in lab[(lab["case_id"] == r.case_id) & (lab["role"] == "layer2")]["acct_id"]:
                i2, o2 = g[g["dst"] == l2], g[g["src"] == l2]
                d2 = (o2["ts"].min() - i2["ts"].min()).total_seconds()
                bad_l2 += not (HOP2_DELAY_S[0] - 0.01 <= d2 <= HOP2_DELAY_S[1] + 0.01)
    check("outbound always after inbound", bad_order == 0, f"{bad_order} bad")
    check("dwell time inside its profile range", bad_dwell == 0, f"{bad_dwell} bad")
    check("90-99% of inbound is forwarded", bad_ratio == 0, f"{bad_ratio} bad")
    check("layer-2 mules forward in 1-5 min", bad_l2 == 0, f"{bad_l2} bad")

    m = sc[sc["kind"] == "mule"]
    t_min, t_max = tx["ts"].min(), tx["ts"].max()
    q1, q2 = tx.loc[~tx["is_injected"], "ts"].quantile([1 / 3, 2 / 3])   # terciles of base TRAFFIC
    third = (m["t0"] > q1).astype(int) + (m["t0"] > q2).astype(int)
    counts = third.value_counts().reindex([0, 1, 2], fill_value=0)
    check("mules follow base traffic over time (each third of traffic has >= 20%)",
          bool((counts >= 0.2 * len(m)).all()), str(counts.to_dict()))

    # ---------------------------------------------------------------- decoys
    print("\n== decoys ==")
    min_delay = {"legit_biz_supplier": 2 * 3600, "legit_aggregator": 2 * 3600, "legit_one_off": 6 * 3600}
    for typ, floor_s in min_delay.items():
        bad = 0
        for r in sc[sc["typology"] == typ].itertuples():
            g = by_case[r.case_id]
            inb, outs = g[g["dst"] == r.primary_acct], g[g["src"] == r.primary_acct]
            if len(outs):
                bad += (outs["ts"].min() - inb["ts"].min()).total_seconds() < floor_s - 0.01
        check(f"{typ}: first outbound >= {floor_s // 3600} h after first inbound", bad == 0, f"{bad} bad")
    bad = 0
    for r in sc[sc["typology"] == "legit_trusted_forwarder"].itertuples():
        g = by_case[r.case_id]
        i = g[g["dst"] == r.primary_acct].sort_values("ts")["ts"].reset_index(drop=True)
        o = g[g["src"] == r.primary_acct].sort_values("ts")["ts"].reset_index(drop=True)
        d = (o - i).dt.total_seconds()
        bad += len(i) != len(o) or not d.between(299.99, 2700.01).all()
    check("trusted_forwarder: repeated, forwards in 5-45 min", bad == 0, f"{bad} bad")

    # ---------------------------------------------------------------- summary
    print("\n== summary ==")
    base_n = int((~tx["is_injected"]).sum())
    mule_tx = int(inj["case_id"].str.startswith("MULE").sum())
    print(f"base transactions     : {base_n:,}")
    print(f"injected mule txns    : {mule_tx:,}")
    print(f"injected decoy txns   : {len(inj) - mule_tx:,}")
    print(f"accounts              : {len(acc):,}")
    print(f"mule accounts         : {int(lab['is_mule'].sum()):,} "
          f"({int((lab['is_mule'] & (lab['source'] == 'ibm')).sum()):,} from IBM patterns, "
          f"{int((lab['is_mule'] & (lab['source'] == 'injected')).sum()):,} injected)")
    print(f"decoy accounts        : {int((lab['label'] == 'decoy').sum()):,}")
    print(f"laundering txn rate   : {tx['is_laundering'].mean():.4%}")
    print(f"time range            : {t_min} -> {t_max}")
    print("electronic-channel mix, base vs injected mules (leakage check; should look broadly alike):")
    elec = ["NEFT", "RTGS", "IMPS", "UPI"]
    b = tx[~tx["is_injected"] & tx["channel"].isin(elec)]["channel"].value_counts(normalize=True)
    mm = inj[inj["case_id"].str.startswith("MULE")]["channel"].value_counts(normalize=True)
    print(pd.concat([b.rename("base"), mm.rename("mule")], axis=1).fillna(0).round(3).to_string())
    print("(mules never use card/cheque/cash: intended, fast transfers are electronic)")

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
