"""Retrieve the most similar past cases for an account.

    python src/similar_cases.py --acct <ACCOUNT_ID>
    python src/similar_cases.py --demo

Importable: find_similar(conn, acct, k=5) -> compact dict (used by the API and the investigation agent).
Uses the account's features from the latest window that contains it (test, then val, then train).
"""
import argparse
import json
import os
from collections import Counter

import psycopg

import embedding as emb

DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
VCAST = os.environ.get("MULEGUARD_VECTOR_TYPE", "vector").split("(")[0]


def find_similar(conn, acct, k=5):
    mean, std = emb.load_scaler()
    for split in ("test", "val", "train"):
        df = emb.fetch_features(conn, split, [acct])
        if len(df):
            break
    else:
        return {"account": acct, "error": "account has no activity in any analysed window"}
    v = emb.vec_text(emb.embed(emb.raw_matrix(df), mean, std)[0])
    rows = conn.execute(f"""SELECT acct_id, case_id, kind, typology, source, summary, 1 - (embedding <=> %s::{VCAST}) AS sim
                            FROM cases WHERE acct_id <> %s ORDER BY embedding <=> %s::{VCAST} LIMIT %s""",
                        (v, acct, v, k)).fetchall()
    nb = [{"case_account": r[0], "case_id": r[1], "kind": r[2], "typology": r[3], "source": r[4],
           "summary": r[5], "similarity": round(float(r[6]), 3)} for r in rows]
    mules = [n for n in nb if n["kind"] == "mule"]
    return {
        "account": acct, "window": split, "this_account": emb.describe(df.iloc[0].to_dict()),
        "neighbours": nb, "mule_share": round(len(mules) / max(len(nb), 1), 2),
        "most_common_mule_typology": (Counter(n["typology"] for n in mules).most_common(1) or [[None]])[0][0],
        "closest_decoy": next((n for n in nb if n["kind"] == "decoy"), None),
    }


def render(r):
    if "error" in r:
        return f"{r['account']}: {r['error']}"
    L = [f"ACCOUNT {r['account']}  (window: {r['window']})", f"  this account {r['this_account']}", "",
         f"SIMILAR PAST CASES - {r['mule_share']:.0%} of the nearest {len(r['neighbours'])} are confirmed mules; "
         f"most common mule type: {r['most_common_mule_typology']}"]
    for n in r["neighbours"]:
        L.append(f"  [{n['kind']:<5}] {n['typology']:<24} sim {n['similarity']:.3f}  ({n['case_id'] or n['source']})")
        L.append(f"          {n['summary']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acct")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    with psycopg.connect(DSN, autocommit=True) as conn:
        acct = a.acct
        if a.demo:
            acct, typ = conn.execute("""SELECT s.primary_acct, s.typology FROM scenarios s, splits p
                WHERE s.kind = 'mule' AND p.name = 'test' AND s.t0 >= p.start_ts AND s.t0 < p.end_ts
                ORDER BY random() LIMIT 1""").fetchone()
            print(f"[demo] picked {acct} (ground truth: {typ})\n")
        if not acct:
            ap.error("give --acct or --demo")
        r = find_similar(conn, acct)
        print(json.dumps(r, indent=2) if a.json else render(r))


if __name__ == "__main__":
    main()
