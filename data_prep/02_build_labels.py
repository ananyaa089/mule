"""Step 2 - turn HI-Small_Patterns.txt into account-level mule labels.

Input : data/raw/HI-Small_Patterns.txt
Output: data/clean/labels_base.parquet   (one row per labelled account)
        data/clean/patterns.parquet      (one row per laundering attempt)

Role rules, per laundering attempt:
  intermediary : account that both receives and sends inside the pattern (the classic mule)
  hub          : if no intermediary exists, the centre of a fan-out (sender) / fan-in (receiver)
  participant  : any other pattern with no intermediary (e.g. bipartite)
"""
import csv
import sys
from collections import Counter

import pandas as pd

from config import CLEAN, RAW_PATTERNS

ROLE_RANK = {"intermediary": 0, "hub": 1, "participant": 2}


def parse(path):
    attempts, cur = [], None
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                name = line.split(" - ", 1)[1] if " - " in line else "UNKNOWN"
                name = name.split(":")[0].strip().lower().replace("-", "_").replace(" ", "_")
                cur = {"type": name, "rows": []}
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if cur is not None:
                    attempts.append(cur)
                cur = None
            elif cur is not None:
                r = next(csv.reader([line]))
                if len(r) < 8:
                    continue
                src, dst = f"{r[1]}_{r[2]}", f"{r[3]}_{r[4]}"
                if src != dst:
                    cur["rows"].append((r[0], src, dst, float(r[7])))
    return attempts


def main() -> None:
    if not RAW_PATTERNS.exists():
        sys.exit(f"Missing {RAW_PATTERNS}. Put HI-Small_Patterns.txt in data/raw/ first.")
    CLEAN.mkdir(parents=True, exist_ok=True)

    attempts = parse(RAW_PATTERNS)
    lab_rows, pat_rows = [], []
    for i, a in enumerate(attempts, 1):
        rows = a["rows"]
        if not rows:
            continue
        case_id = f"IBM-{i:05d}"
        srcs, dsts = Counter(r[1] for r in rows), Counter(r[2] for r in rows)
        accounts = set(srcs) | set(dsts)
        inter = set(srcs) & set(dsts)

        roles = {}
        if inter:
            roles = {x: "intermediary" for x in inter}
        elif a["type"] == "fan_out":
            roles = {srcs.most_common(1)[0][0]: "hub"}
        elif a["type"] == "fan_in":
            roles = {dsts.most_common(1)[0][0]: "hub"}
        else:
            roles = {x: "participant" for x in accounts}

        for acct, role in roles.items():
            lab_rows.append((acct, "mule", role, a["type"], case_id, "ibm"))
        ts = pd.to_datetime([r[0] for r in rows], format="%Y/%m/%d %H:%M")
        pat_rows.append((case_id, a["type"], len(rows), len(accounts),
                         ts.min(), ts.max(), sum(r[3] for r in rows)))

    labels = pd.DataFrame(lab_rows, columns=["acct_id", "label", "role", "typology", "case_id", "source"])
    labels["rank"] = labels["role"].map(ROLE_RANK)
    labels = (labels.sort_values(["acct_id", "rank", "case_id"])
                    .drop_duplicates("acct_id").drop(columns="rank").reset_index(drop=True))
    patterns = pd.DataFrame(pat_rows, columns=["case_id", "typology", "n_txns", "n_accounts",
                                               "start_ts", "end_ts", "total_amount"])

    print(f"laundering attempts parsed : {len(patterns):,}")
    print(patterns["typology"].value_counts().to_string())
    print(f"\nlabelled accounts          : {len(labels):,}")
    print(labels["role"].value_counts().to_string())

    labels.to_parquet(CLEAN / "labels_base.parquet", index=False)
    patterns.to_parquet(CLEAN / "patterns.parquet", index=False)
    print(f"\nwrote {CLEAN / 'labels_base.parquet'} and patterns.parquet")


if __name__ == "__main__":
    main()
