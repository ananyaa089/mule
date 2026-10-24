"""Step 3 - inject synthetic mule scenarios and look-alike decoys.

Input : data/clean/transactions_base.parquet, labels_base.parquet
Output: data/final/transactions.parquet   (base + injected, time-sorted, txn_id assigned)
        data/final/scenarios.parquet      (one row per injected scenario)
        data/clean/labels_injected.parquet

Mule scenarios  : Rs 3-8 lakh arrives, 90-99% leaves within 20-120 s (some 2-10 min),
                  optionally split over several recipients and forwarded again (layering).
Decoy scenarios : legitimate accounts with big flows but human-scale timing, so false
                  positives are measurable.

Injected rows only use *existing* accounts, so there is no giveaway in account ids.
Mules/decoys are drawn from low-activity accounts so "low activity" alone is not a signal.
"""
import sys

import numpy as np
import pandas as pd

from config import (CLEAN, DECOY_KIND_WEIGHTS, FAST_DWELL_S, FAST_PROFILE_PROB, FINAL,
                    FORWARD_FRACTION, HOP2_DELAY_S, LOW_ACTIVITY_MAX, MEDIUM_DWELL_S,
                    MULE_AMOUNT, MULE_KIND_WEIGHTS, N_DECOY_SCENARIOS, N_MULE_SCENARIOS, SEED)

HOUR, DAY = 3600.0, 86400.0


def main() -> None:
    src_file = CLEAN / "transactions_base.parquet"
    if not src_file.exists():
        sys.exit("Run 01_clean_base.py and 02_build_labels.py first.")
    FINAL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)

    base = pd.read_parquet(src_file)
    lab_ibm = pd.read_parquet(CLEAN / "labels_base.parquet")
    t_min, t_max = base["ts"].min(), base["ts"].max()
    span_s = (t_max - t_min).total_seconds()
    margin = min(DAY, 0.05 * span_s)
    print(f"base time range: {t_min} -> {t_max}  ({span_s / DAY:.1f} days)")

    # ------------------------------------------------------------ account pools
    activity = pd.concat([base["src"], base["dst"]], ignore_index=True).value_counts()
    not_ibm = ~activity.index.isin(lab_ibm["acct_id"])
    low = activity.index[(activity.values <= LOW_ACTIVITY_MAX) & not_ibm].to_numpy()
    peers = activity.index[(activity.values > LOW_ACTIVITY_MAX) & not_ibm].to_numpy()
    rng.shuffle(low)
    need = N_MULE_SCENARIOS * 3 + N_DECOY_SCENARIOS
    print(f"low-activity pool: {len(low):,}   peer pool: {len(peers):,}")
    if len(low) < need or len(peers) < 50:
        sys.exit(f"Account pools too small (need >= {need:,} low-activity). "
                 f"Raise LOW_ACTIVITY_MAX in config.py or lower the scenario counts.")
    low_i = 0

    def take_labeled() -> str:
        nonlocal low_i
        a = low[low_i]
        low_i += 1
        return a

    def take_peers(k: int, exclude=()) -> list:
        out = []
        while len(out) < k:
            a = peers[rng.integers(len(peers))]
            if a not in out and a not in exclude:
                out.append(a)
        return out

    # ------------------------------------------------------------ helpers
    rows, labels, scen = [], [], []

    def add(ts, s, d, amt, ch, laundering, cid):
        rows.append((ts, s, d, float(amt), ch, int(laundering), cid))

    def at(t0, seconds):
        return t0 + pd.Timedelta(milliseconds=int(round(seconds * 1000)))

    def rand_ts(lo, hi):
        secs = rng.uniform(0, max((hi - lo).total_seconds(), 1.0))
        return at(lo, secs)

    # Channel for injected rows is sampled from the BASE data's own electronic-channel mix
    # for the same amount range. A hand-picked mix (e.g. mostly IMPS/UPI) would make the
    # channel column reveal which rows are injected.
    elec = base[base["channel"].isin(["NEFT", "RTGS", "IMPS", "UPI"])]
    edges = np.array([0, 10_000, 100_000, 200_000, 500_000, np.inf])
    bin_idx = np.clip(np.searchsorted(edges, elec["amount"].to_numpy(), side="right") - 1, 0, 4)
    chan_tab = {}
    for b in range(5):
        vc = elec["channel"][bin_idx == b].value_counts()
        if len(vc):
            chan_tab[b] = (vc.index.to_numpy(), (vc / vc.sum()).to_numpy())
    if not chan_tab:
        sys.exit("Base data has no NEFT/RTGS/IMPS/UPI rows to copy channel mix from.")

    def pick_channel(amt):
        b = int(np.clip(np.searchsorted(edges, amt, side="right") - 1, 0, 4))
        key = min(chan_tab, key=lambda x: abs(x - b))      # nearest bin that has data
        names, probs = chan_tab[key]
        return str(rng.choice(names, p=probs))

    in_channel = out_channel = pick_channel

    def split(total, n, alpha=3.0):
        return np.floor(rng.dirichlet(np.full(n, alpha)) * total)

    def amount_in():
        return float(rng.integers(MULE_AMOUNT[0], MULE_AMOUNT[1] + 1))

    # ------------------------------------------------------------ mule scenarios
    kinds = list(MULE_KIND_WEIGHTS)
    kp = np.array(list(MULE_KIND_WEIGHTS.values())) / sum(MULE_KIND_WEIGHTS.values())
    mule_lo, mule_hi = t_min + pd.Timedelta(seconds=margin), t_max - pd.Timedelta(seconds=margin + 1200)

    for k in range(1, N_MULE_SCENARIOS + 1):
        cid, kind = f"MULE-{k:04d}", str(rng.choice(kinds, p=kp))
        profile = "fast" if rng.random() < FAST_PROFILE_PROB else "medium"
        dlo, dhi = FAST_DWELL_S if profile == "fast" else MEDIUM_DWELL_S
        t0 = rand_ts(mule_lo, mule_hi)
        mule, source = take_labeled(), take_peers(1)[0]
        amt = amount_in()
        add(t0, source, mule, amt, in_channel(amt), 1, cid)
        labels.append((mule, "mule", "primary", kind, cid, "injected"))

        n_out = {"single_pass": 1, "fan_out": int(rng.integers(3, 11)),
                 "two_hop": int(rng.integers(1, 4))}[kind]
        pieces = split(amt * rng.uniform(*FORWARD_FRACTION), n_out)
        delays = np.sort(rng.uniform(dlo, dhi, n_out))
        if kind == "two_hop":
            recipients = [take_labeled() for _ in range(n_out)]
        else:
            recipients = take_peers(n_out, exclude=(source,))

        n_txn = 1 + n_out
        for piece, d, r in zip(pieces, delays, recipients):
            t1 = at(t0, d)
            add(t1, mule, r, piece, out_channel(piece), 1, cid)
            if kind == "two_hop":
                labels.append((r, "mule", "layer2", kind, cid, "injected"))
                n2 = int(rng.integers(1, 4))
                pieces2 = split(piece * rng.uniform(*FORWARD_FRACTION), n2)
                finals = take_peers(n2, exclude=(source,))
                for p2, f in zip(pieces2, finals):
                    add(at(t1, rng.uniform(*HOP2_DELAY_S)), r, f, p2, out_channel(p2), 1, cid)
                    n_txn += 1
        scen.append(dict(case_id=cid, kind="mule", typology=kind, primary_acct=mule,
                         t0=t0, amount=amt, n_txns=n_txn, dwell_profile=profile))

    # ------------------------------------------------------------ decoy scenarios
    dkinds = list(DECOY_KIND_WEIGHTS)
    dp = np.array(list(DECOY_KIND_WEIGHTS.values())) / sum(DECOY_KIND_WEIGHTS.values())
    unit = min(1.0, span_s / (8 * DAY))          # shrink long delays only if the data span is short

    def pick_t0(length_s):
        hi = t_max - pd.Timedelta(seconds=length_s + margin)
        lo = t_min + pd.Timedelta(seconds=margin)
        return rand_ts(lo, max(hi, lo))

    for k in range(1, N_DECOY_SCENARIOS + 1):
        cid, kind = f"DECOY-{k:04d}", str(rng.choice(dkinds, p=dp))
        acct = take_labeled()
        labels.append((acct, "decoy", kind, f"legit_{kind}", cid, "injected"))
        n_txn, amt_total = 0, 0.0

        if kind == "biz_supplier":
            amt = amount_in()
            k_in, n_out = int(rng.integers(1, 4)), int(rng.integers(3, 13))
            in_parts = split(amt, k_in, 5.0)
            in_off = np.sort(rng.uniform(0, 1800, k_in))
            lo_d, hi_d = 2.5 * HOUR, max(4 * DAY * unit, 3.5 * HOUR)
            out_off = np.sort(rng.uniform(lo_d, hi_d, n_out))
            out_parts = split(amt * rng.uniform(0.70, 0.99), n_out, 2.0)
            t0 = pick_t0(out_off.max())
            for p, o, s in zip(in_parts, in_off, take_peers(k_in)):
                add(at(t0, o), s, acct, p, in_channel(p), 0, cid)
            for p, o, r in zip(out_parts, out_off, take_peers(n_out)):
                add(at(t0, o), acct, r, p, out_channel(p), 0, cid)
            n_txn, amt_total = k_in + n_out, amt

        elif kind == "aggregator":
            senders, settle = take_peers(int(rng.integers(5, 16))), take_peers(1)[0]
            days = max(1, min(int(rng.integers(2, 5)), int(span_s // DAY) - 1))
            t0 = pick_t0(days * DAY)
            for day in range(days):
                m = int(rng.integers(15, 41))
                offs = np.sort(rng.uniform(0, 12 * HOUR, m)) + day * DAY
                amts = rng.integers(20_000, 100_001, m).astype(float)
                for o, a in zip(offs, amts):
                    add(at(t0, o), senders[int(rng.integers(len(senders)))], acct, a,
                        out_channel(a), 0, cid)
                out = float(np.floor(amts.sum() * rng.uniform(0.97, 0.995)))
                add(at(t0, day * DAY + rng.uniform(14 * HOUR, 20 * HOUR)), acct, settle, out,
                    out_channel(out), 0, cid)
                n_txn += m + 1
                amt_total += amts.sum()

        elif kind == "one_off":
            amt = amount_in()
            t0 = pick_t0(3 * DAY * unit)
            add(t0, take_peers(1)[0], acct, amt, in_channel(amt), 0, cid)
            n_txn, amt_total = 1, amt
            if rng.random() < 0.7:
                n_out = int(rng.integers(1, 3))
                for p, r in zip(split(amt * rng.uniform(0.2, 0.6), n_out),
                                take_peers(n_out)):
                    add(at(t0, rng.uniform(6 * HOUR, max(3 * DAY * unit, 7 * HOUR))), acct, r, p,
                        out_channel(p), 0, cid)
                    n_txn += 1

        else:  # trusted_forwarder: same two counterparties, repeated, quick pass-through
            s, d = take_peers(2)
            reps = int(rng.integers(3, 6))
            gaps = rng.uniform(0.5 * DAY, 1.5 * DAY, reps) * min(1.0, unit * 2)
            offs = np.concatenate([[0.0], np.cumsum(gaps[:-1])])
            t0 = pick_t0(offs[-1] + HOUR)
            for o in offs:
                amt = amount_in()
                add(at(t0, o), s, acct, amt, in_channel(amt), 0, cid)
                fwd = float(np.floor(amt * rng.uniform(0.985, 0.998)))
                add(at(t0, o + rng.uniform(300, 2700)), acct, d, fwd, out_channel(fwd), 0, cid)
                n_txn += 2
                amt_total += amt

        scen.append(dict(case_id=cid, kind="decoy", typology=f"legit_{kind}", primary_acct=acct,
                         t0=t0, amount=amt_total, n_txns=n_txn, dwell_profile=None))

    # ------------------------------------------------------------ merge + write
    inj = pd.DataFrame(rows, columns=["ts", "src", "dst", "amount", "channel", "is_laundering", "case_id"])
    inj["channel_orig"] = None
    inj["is_injected"] = True
    base["is_injected"] = False
    base["case_id"] = None

    tx = pd.concat([base, inj], ignore_index=True)
    tx = tx.sort_values("ts", kind="mergesort").reset_index(drop=True)
    tx["txn_id"] = np.arange(len(tx), dtype="int64")
    tx["currency"] = "INR"
    tx = tx[["txn_id", "ts", "src", "dst", "amount", "currency", "channel", "channel_orig",
             "is_laundering", "is_injected", "case_id"]]

    tx.to_parquet(FINAL / "transactions.parquet", index=False)
    pd.DataFrame(scen).to_parquet(FINAL / "scenarios.parquet", index=False)
    pd.DataFrame(labels, columns=["acct_id", "label", "role", "typology", "case_id", "source"]) \
        .to_parquet(CLEAN / "labels_injected.parquet", index=False)

    n_mule_tx = int(inj["case_id"].str.startswith("MULE").sum())
    print(f"\ninjected mule scenarios  : {N_MULE_SCENARIOS}  ({n_mule_tx:,} txns)")
    print(f"injected decoy scenarios : {N_DECOY_SCENARIOS}  ({len(inj) - n_mule_tx:,} txns)")
    print(f"total transactions       : {len(tx):,}")
    print(f"wrote {FINAL / 'transactions.parquet'}")


if __name__ == "__main__":
    main()
