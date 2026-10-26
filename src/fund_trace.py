"""Fund tracing for one account: where did the money come from, and where did it go?

    python src/fund_trace.py --acct 011_8000ECA90          # trace the account's most suspicious inbound transfer
    python src/fund_trace.py --acct ACCT --t0 "2022-09-10 14:03:22" --amount 500000
    python src/fund_trace.py --demo                        # picks an injected mule from the test period

Also importable (the API and the investigation agent use trace_account()):
    from fund_trace import trace_account
    result = trace_account(conn, "011_8000ECA90")          # plain dict, JSON-safe, deliberately compact

It uses only transactions + ml_scores + accounts. It never reads ground-truth labels (the --demo picker is the
one exception, and only to choose an example).
"""
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
TOP_PER_DEPTH = 8          # keeps the result small enough for an LLM with a tight token budget


def install(conn):
    conn.execute((ROOT / "sql" / "05_trace.sql").read_text())


def threshold():
    try:
        meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
        return float(meta["results"][meta["main_model"]]["threshold"])
    except Exception:
        return None


def find_trigger(conn, acct):
    """The inbound transfer that looks most like pass-through: highest share forwarded within 15 min
    (ties -> larger amount), among the account's 25 largest inbound transfers."""
    row = conn.execute("""
        WITH cand AS (SELECT txn_id, ts, src, amount, channel FROM transactions WHERE dst = %s
                      ORDER BY amount DESC LIMIT 25),
        scored AS (
            SELECT c.*, (SELECT coalesce(sum(o.amount), 0) FROM transactions o
                          WHERE o.src = %s AND o.ts > c.ts AND o.ts <= c.ts + interval '15 minutes') AS fwd
            FROM cand c)
        SELECT txn_id, ts, src, amount, channel, fwd FROM scored
        ORDER BY least(fwd / amount, 1) DESC, amount DESC LIMIT 1""", (acct, acct)).fetchone()
    return None if row is None else dict(zip(("txn_id", "ts", "src", "amount", "channel", "fwd15"), row))


def _scores(conn, accts):
    rows = conn.execute("""SELECT a.acct_id, a.status, a.bank, (SELECT max(score) FROM ml_scores m WHERE m.acct_id = a.acct_id)
                           FROM accounts a WHERE a.acct_id = ANY(%s)""", (list(accts),)).fetchall()
    return {r[0]: {"status": r[1], "bank": r[2], "ml_score": None if r[3] is None else round(r[3], 4)} for r in rows}


def trace_account(conn, acct, t0=None, seed_amount=None, horizon_min=30, lookback_h=24, fwd_depth=3, back_depth=2):
    install(conn)
    trig = None
    if t0 is None or seed_amount is None:
        trig = find_trigger(conn, acct)
        if trig is None:
            return {"account": acct, "error": "account has no inbound transfers"}
        t0, seed_amount = trig["ts"], float(trig["amount"])
    elif isinstance(t0, str):
        t0 = datetime.fromisoformat(t0)
    thr = threshold()

    fwd = conn.execute("SELECT * FROM trace_forward(%s,%s,%s,%s,%s)", (
        acct, t0, seed_amount, timedelta(minutes=horizon_min), fwd_depth)).fetchall()
    bwd = conn.execute("SELECT * FROM trace_backward(%s,%s,%s,%s,%s)", (
        acct, t0, seed_amount, timedelta(hours=lookback_h), back_depth)).fetchall()
    cols = ("depth", "txn_id", "ts", "src", "dst", "amount", "link")
    fwd = [dict(zip(cols, r)) for r in fwd]
    bwd = [dict(zip(cols, r)) for r in bwd]

    nodes = {acct} | {r["dst"] for r in fwd} | {r["src"] for r in bwd}
    meta = _scores(conn, nodes)

    def hop(r, forward):
        other = r["dst"] if forward else r["src"]
        return {"depth": r["depth"], "from": r["src"], "to": r["dst"], "amount": round(r["amount"], 2),
                "secs_after_trigger": round((r["ts"] - t0).total_seconds(), 1), "txn_id": r["txn_id"],
                "other_ml_score": meta.get(other, {}).get("ml_score")}

    def top(rows, forward):
        out = []
        for d in sorted({r["depth"] for r in rows}):
            lvl = sorted((r for r in rows if r["depth"] == d), key=lambda r: -r["amount"])
            out.append({"depth": d, "n_txns": len(lvl), "total_amount": round(sum(r["amount"] for r in lvl), 2),
                        "shown": [hop(r, forward) for r in lvl[:TOP_PER_DEPTH]]})
        return out

    d1 = [r for r in fwd if r["depth"] == 1]
    sent_ids = {r["src"] for r in fwd}
    finals = {}
    for r in fwd:
        if r["dst"] not in sent_ids:                      # received but did not pass on within the trace
            finals[r["dst"]] = finals.get(r["dst"], 0) + r["amount"]
    out1 = sum(r["amount"] for r in d1)

    suspects = []
    for a in nodes - {acct}:
        m = meta.get(a, {})
        role = "source" if any(r["src"] == a for r in bwd) and a not in {r["dst"] for r in fwd} else "recipient"
        suspects.append({"account": a, "role": role, "ml_score": m.get("ml_score"), "status": m.get("status"),
                         "above_threshold": None if (thr is None or m.get("ml_score") is None) else m["ml_score"] >= thr})
    suspects.sort(key=lambda s: -(s["ml_score"] or 0))

    return {
        "account": acct,
        "account_info": meta.get(acct, {}),
        "ml_threshold": thr,
        "trigger": {"txn_id": None if trig is None else trig["txn_id"], "ts": str(t0), "amount": round(seed_amount, 2),
                    "from": None if trig is None else trig["src"], "channel": None if trig is None else trig["channel"]},
        "forward": {
            "forwarded_first_hop": round(out1, 2), "share_of_trigger": round(out1 / seed_amount, 3) if seed_amount else None,
            "first_out_secs": min((round((r["ts"] - t0).total_seconds(), 1) for r in d1), default=None),
            "last_out_secs": max((round((r["ts"] - t0).total_seconds(), 1) for r in d1), default=None),
            "n_first_hop_recipients": len({r["dst"] for r in d1}),
            "max_depth_reached": max((r["depth"] for r in fwd), default=0),
            "hops": top(fwd, True),
            "end_of_trace": [{"account": a, "received": round(v, 2), "ml_score": meta.get(a, {}).get("ml_score")}
                             for a, v in sorted(finals.items(), key=lambda kv: -kv[1])[:TOP_PER_DEPTH]],
            "truncated": len(fwd) >= 300},
        "backward": {"n_sources": len({r["src"] for r in bwd if r["depth"] == 1}), "hops": top(bwd, False),
                     "truncated": len(bwd) >= 300},
        "suspects": suspects[:10],
    }


def render(res):
    if "error" in res:
        return f"{res['account']}: {res['error']}"
    t, f, b = res["trigger"], res["forward"], res["backward"]
    L = [f"ACCOUNT {res['account']}  (status {res['account_info'].get('status')}, ML score {res['account_info'].get('ml_score')},"
         f" threshold {res['ml_threshold']})",
         f"TRIGGER  {t['ts']}  Rs {t['amount']:,.0f} received from {t['from']} via {t['channel']}", "",
         f"WHERE IT WENT  - {f['share_of_trigger']:.0%} of it left in the first hop: {f['n_first_hop_recipients']} recipient(s), "
         f"first out after {f['first_out_secs']}s, last after {f['last_out_secs']}s, depth {f['max_depth_reached']}"]
    for lvl in f["hops"]:
        L.append(f"  hop {lvl['depth']}: {lvl['n_txns']} txns, Rs {lvl['total_amount']:,.0f}")
        for h in lvl["shown"]:
            L.append(f"     +{h['secs_after_trigger']:>7}s  {h['from']} -> {h['to']}  Rs {h['amount']:,.0f}  (receiver score {h['other_ml_score']})")
    L.append("  money ends at: " + (", ".join(f"{e['account']} Rs {e['received']:,.0f}" for e in f["end_of_trace"]) or "n/a"))
    L += ["", f"WHERE IT CAME FROM - {b['n_sources']} direct source(s)"]
    for lvl in b["hops"]:
        L.append(f"  step {lvl['depth']} back: {lvl['n_txns']} txns, Rs {lvl['total_amount']:,.0f}")
        for h in lvl["shown"][:5]:
            L.append(f"     {h['secs_after_trigger']:>9}s  {h['from']} -> {h['to']}  Rs {h['amount']:,.0f}  (sender score {h['other_ml_score']})")
    L += ["", "ACCOUNTS TO REVIEW (by ML score):"]
    for s in res["suspects"][:6]:
        L.append(f"  {s['account']:<22} {s['role']:<10} score {s['ml_score']}  {'ABOVE threshold' if s['above_threshold'] else ''}  [{s['status']}]")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acct")
    ap.add_argument("--t0")
    ap.add_argument("--amount", type=float)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    with psycopg.connect(DSN, autocommit=True) as conn:
        acct = a.acct
        if a.demo:
            acct, case, typ = conn.execute("""SELECT s.primary_acct, s.case_id, s.typology FROM scenarios s, splits p
                WHERE s.kind = 'mule' AND p.name = 'test' AND s.t0 >= p.start_ts AND s.t0 < p.end_ts
                ORDER BY random() LIMIT 1""").fetchone()
            print(f"[demo] picked {acct} ({case}, ground truth: {typ})\n")
        if not acct:
            ap.error("give --acct or --demo")
        res = trace_account(conn, acct, a.t0, a.amount)
        print(json.dumps(res, indent=2, default=str) if a.json else render(res))


if __name__ == "__main__":
    main()
