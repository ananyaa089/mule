"""MuleGuard investigation agent: one account in, one explainable report out.

    python src/agent.py --demo                 # investigate a random injected mule from the test period
    python src/agent.py --demo decoy           # ...or a legitimate look-alike (should NOT come out high-risk)
    python src/agent.py --acct <ACCOUNT_ID>
    python src/agent.py --acct <ACCOUNT_ID> --no-llm      # deterministic report, no API call
    python src/agent.py --acct <ACCOUNT_ID> --json

Design (important for explainability):
  1. EVIDENCE is gathered by code from the real tools: XGBoost score, anomaly score, fund trace, similar cases.
  2. RISK LEVEL, recommended action and freeze candidates are decided by FIXED RULES (assess()), never by the LLM.
  3. The LLM only WRITES the narrative (summary, key patterns, caveats) from that evidence.
  4. Every number in the narrative is checked against the evidence; anything not found is listed as unverified.
  5. If there is no API key or the call fails, a template narrative is used, so the system always produces a report.
  The report only ever RECOMMENDS human review; freezing is a separate, confirmed action.

LLM settings (any OpenAI-compatible provider; defaults are for Groq):
  GROQ_API_KEY or LLM_API_KEY      the key
  LLM_BASE_URL   default https://api.groq.com/openai/v1
  LLM_MODEL      default llama-3.3-70b-versatile
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import psycopg

import fund_trace
import similar_cases

DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

# ---- fixed decision rules (change here, not in a prompt) -------------------------------------------------------
FAST_SHARE, FAST_SECS = 0.80, 120          # >= 80% of the trigger transfer left within 2 minutes -> "fast pass-through"
SLOW_SECS = 900                            # ... within 15 minutes -> "pass-through"
ANOMALY_HIGH, ANOMALY_MED = 0.99, 0.95     # percentile among normal accounts


class LLMUnavailable(Exception):
    pass


# ------------------------------------------------------------------------------------------------ evidence
def profile(conn, acct):
    row = conn.execute("""SELECT m.score, m.split FROM ml_scores m WHERE m.acct_id = %s
                          ORDER BY CASE m.split WHEN 'test' THEN 0 WHEN 'val' THEN 1 ELSE 2 END LIMIT 1""", (acct,)).fetchone()
    acc = conn.execute("SELECT status, bank, kyc_level, account_type FROM accounts WHERE acct_id = %s", (acct,)).fetchone()
    return {"status": acc[0] if acc else None, "bank": acc[1] if acc else None,
            "ml_score": None if row is None else round(float(row[0]), 4), "window": None if row is None else row[1],
            "ml_threshold": fund_trace.threshold()}


def gather(conn, acct, t0=None, amount=None):
    ev = {"account": acct, "profile": profile(conn, acct)}
    ev["trace"] = fund_trace.trace_account(conn, acct, t0, amount)
    ev["similar"] = similar_cases.find_similar(conn, acct)
    try:
        import anomaly                                   # needs torch; the report still works without it
        ev["anomaly"] = anomaly.score_account(conn, acct)
    except Exception as e:                               # noqa: BLE001
        ev["anomaly"] = {"account": acct, "error": f"anomaly model unavailable ({e.__class__.__name__})"}
    return ev


# ------------------------------------------------------------------------------------------------ rules
def assess(ev):
    p, tr, sim, an = ev["profile"], ev["trace"], ev["similar"], ev["anomaly"]
    thr, ml = p.get("ml_threshold"), p.get("ml_score")
    ml_hi = ml is not None and thr is not None and ml >= thr
    anom = an.get("anomaly_pct")
    fwd = tr.get("forward") or {}
    share, first = fwd.get("share_of_trigger") or 0, fwd.get("first_out_secs")
    fast = share >= FAST_SHARE and first is not None and first <= FAST_SECS
    passthru = share >= FAST_SHARE and first is not None and first <= SLOW_SECS
    layered = (fwd.get("max_depth_reached") or 0) >= 2
    flagged_down = [s for s in tr.get("suspects", []) if s.get("above_threshold") and s["role"] == "recipient"]

    reasons = []
    if ml is not None and thr is not None:
        reasons.append(f"Supervised model score {ml} is {'above' if ml_hi else 'below'} the alert threshold {round(thr, 4)}.")
    if tr.get("trigger") and fwd:
        t = tr["trigger"]
        reasons.append(f"Transfer of Rs {t['amount']:,.0f} (txn {t['txn_id']}) was {share:.0%} forwarded in the first hop"
                       + (f", first outflow {first} seconds after receipt" if first is not None else "")
                       + (" (fast pass-through)." if fast else "."))
    if layered:
        reasons.append(f"Funds moved on through {fwd['max_depth_reached']} hops"
                       + (f"; {len(flagged_down)} downstream account(s) also score above threshold." if flagged_down else "."))
    if anom is not None:
        reasons.append(f"Behaviour is more unusual than {anom:.1%} of ordinary accounts (anomaly percentile {anom}).")
    if sim.get("neighbours"):
        reasons.append(f"{sim['mule_share']:.0%} of the {len(sim['neighbours'])} most similar past cases are confirmed mule cases "
                       f"(context only, not independent evidence).")

    if ml_hi and (fast or (anom or 0) >= ANOMALY_HIGH):
        level = "HIGH"
    elif ml_hi or (fast and ((anom or 0) >= ANOMALY_MED or flagged_down)) or (flagged_down and passthru):
        level = "MEDIUM"
    else:
        level = "LOW"

    cands = []
    if level == "HIGH":
        cands = [ev["account"]] + [s["account"] for s in flagged_down]
    review = [s["account"] for s in tr.get("suspects", []) if s["account"] not in cands and (s.get("ml_score") or 0) >= 0.3][:5]
    action = {"HIGH": "Recommend a temporary freeze of the listed accounts pending HUMAN review. No automatic action is taken.",
              "MEDIUM": "Recommend manual review within 24 hours and enhanced monitoring; do not freeze on this evidence alone.",
              "LOW": "No action recommended; keep under routine monitoring."}[level]
    decoy = sim.get("closest_decoy")
    cav = []
    if decoy:
        cav.append(f"Among similar past cases is a legitimate look-alike ({decoy['typology']}, similarity {decoy['similarity']}); "
                   f"rule out a business or aggregator pattern before acting.")
    if ev["profile"].get("status") == "frozen":
        cav.append("The account is already frozen.")
    return {"risk_level": level, "reasons": reasons, "recommended_action": action,
            "freeze_candidates": cands, "review_accounts": review, "rule_caveats": cav}


# ------------------------------------------------------------------------------------------------ narrative
def prompt_view(ev, a):
    """Compact evidence for the LLM. Keeps the prompt to roughly 2-3k tokens (Groq free tier is ~8k tokens/minute)."""
    tr = ev["trace"]
    f, b = tr.get("forward", {}), tr.get("backward", {})
    hop = lambda h: {"txn": h["txn_id"], "from": h["from"], "to": h["to"], "rs": h["amount"], "secs": h["secs_after_trigger"],
                     "receiver_score": h["other_ml_score"]}
    sim = ev["similar"]
    return {
        "account": ev["account"], "risk_level_decided_by_rules": a["risk_level"], "scores": {
            "ml_score": ev["profile"].get("ml_score"), "ml_threshold": ev["profile"].get("ml_threshold"),
            "anomaly_pct": ev["anomaly"].get("anomaly_pct")},
        "anomaly_top_deviations": ev["anomaly"].get("top_deviations", [])[:3],
        "trigger": tr.get("trigger"),
        "where_it_went": {"share_forwarded_first_hop": f.get("share_of_trigger"), "first_out_secs": f.get("first_out_secs"),
                          "last_out_secs": f.get("last_out_secs"), "recipients_first_hop": f.get("n_first_hop_recipients"),
                          "max_depth": f.get("max_depth_reached"),
                          "hops": [{"depth": l["depth"], "n": l["n_txns"], "total_rs": l["total_amount"],
                                    "top": [hop(h) for h in l["shown"][:4]]} for l in f.get("hops", [])],
                          "end_of_trace": f.get("end_of_trace", [])[:4]},
        "where_it_came_from": {"direct_sources": b.get("n_sources"),
                               "top": [hop(h) for l in b.get("hops", [])[:1] for h in l["shown"][:3]]},
        "accounts_above_threshold": a["freeze_candidates"],
        "similar_past_cases": {"mule_share": sim.get("mule_share"), "top_type": sim.get("most_common_mule_typology"),
                               "nearest": [{"type": n["typology"], "kind": n["kind"], "similarity": n["similarity"]}
                                           for n in sim.get("neighbours", [])[:3]],
                               "closest_decoy": None if not sim.get("closest_decoy") else
                               {"type": sim["closest_decoy"]["typology"], "similarity": sim["closest_decoy"]["similarity"]}},
        "rule_findings": a["reasons"],
    }


SYSTEM = """You are a financial-crime analyst assistant writing an explainable investigation note about ONE bank account.
You are given JSON evidence computed by other systems. The risk level and recommended action were already decided by fixed rules: never change, contradict, or soften them.
Rules:
- Use only facts and numbers present in the evidence. Write amounts and times exactly as given (for example "Rs 378,015", "72.2 seconds"); do not convert units or do arithmetic.
- Refer to accounts and transactions by their IDs, and cite the transaction id (txn) when you describe a transfer.
- Say "behaviour consistent with" rather than accusing anyone. This is a prototype on synthetic data; a human makes the final decision.
- The share of similar past cases that are mules is context, not independent proof.
- Reply with ONLY a JSON object, no markdown, in exactly this shape:
{"summary": "3-4 sentences", "key_patterns": [{"pattern": "short name", "evidence": "one sentence citing ids and numbers"}], "caveats": ["short caveat"]}
Give 2-4 key_patterns and 1-3 caveats."""


def llm_chat(messages, model=None, max_tokens=900, retries=3):
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMUnavailable("no API key set (GROQ_API_KEY or LLM_API_KEY)")
    body = json.dumps({"model": model or LLM_MODEL, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{LLM_BASE.rstrip('/')}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "muleguard/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                wait = float(e.headers.get("Retry-After", 2 ** (attempt + 2)))
                time.sleep(min(wait, 30))
                continue
            raise LLMUnavailable(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise LLMUnavailable(f"network error: {e}") from e
    raise LLMUnavailable("retries exhausted")


def parse_json(text):
    s, e = text.find("{"), text.rfind("}")
    d = json.loads(text[s:e + 1])
    if not isinstance(d.get("summary"), str) or not isinstance(d.get("key_patterns"), list):
        raise ValueError("missing fields")
    d["key_patterns"] = [{"pattern": str(k.get("pattern", "")), "evidence": str(k.get("evidence", ""))}
                         for k in d["key_patterns"] if isinstance(k, dict)][:4]
    d["caveats"] = [str(c) for c in d.get("caveats", [])][:3]
    return d


def template_narrative(ev, a):
    tr = ev["trace"]
    f, t = tr.get("forward", {}), tr.get("trigger", {})
    pats = []
    share, first = f.get("share_of_trigger"), f.get("first_out_secs")
    if share is not None and share >= 0.5 and first is not None:
        pats.append({"pattern": "Rapid pass-through", "evidence":
                     f"Account {ev['account']} received Rs {t['amount']:,.0f} (txn {t['txn_id']}) and forwarded {share:.0%} of it, "
                     f"first outflow {first} seconds after receipt."})
    elif share is not None:
        pats.append({"pattern": "No significant pass-through", "evidence":
                     f"Account {ev['account']} received Rs {t['amount']:,.0f} (txn {t['txn_id']}) and forwarded only {share:.0%} of it "
                     f"within the trace window."})
    if (f.get("max_depth_reached") or 0) >= 2:
        pats.append({"pattern": "Layering", "evidence": f"Funds continued through {f['max_depth_reached']} hops to "
                     f"{len(f.get('end_of_trace', []))} end-of-trace account(s)."})
    if ev["anomaly"].get("anomaly_pct") is not None:
        d = ev["anomaly"].get("top_deviations", [])
        pats.append({"pattern": "Unusual behaviour", "evidence": f"Anomaly percentile {ev['anomaly']['anomaly_pct']}"
                     + (f"; most deviating feature {d[0]['feature']} (z {d[0]['z_vs_normal']})." if d else ".")})
    summary = f"Account {ev['account']} is rated {a['risk_level']} risk by the rule engine. " + " ".join(a["reasons"][:3])
    return {"summary": summary, "key_patterns": pats[:4], "caveats": ["Narrative generated from a template (no language model used)."]}


def _numbers(text):
    text = re.sub(r"\b\w*_\w+\b", " ", text)                       # drop account ids like 0327897_80AF7EEC0
    out = []
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        try:
            out.append(float(m.group().replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def _walk_numbers(o, acc):
    if isinstance(o, bool) or o is None:
        return
    if isinstance(o, (int, float)):
        acc.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            _walk_numbers(v, acc)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _walk_numbers(v, acc)
    elif isinstance(o, str):
        acc.extend(_numbers(o))


def verify_numbers(narr, view):
    """Numbers in the narrative that cannot be found in the evidence (tolerating rounding and x100 for percentages)."""
    allowed = []
    _walk_numbers(view, allowed)
    allowed += [x * 100 for x in allowed if 0 <= x <= 1]
    text = " ".join([narr["summary"]] + [f"{k['pattern']} {k['evidence']}" for k in narr["key_patterns"]] + narr.get("caveats", []))
    bad = []
    for n in _numbers(text):
        if n <= 5 and float(n).is_integer():                       # ordinals / small counts: too noisy to check
            continue
        if not any(abs(n - b) <= 0.006 * max(abs(b), 1.0) + 0.05 for b in allowed):
            bad.append(n)
    return sorted(set(bad))


def write_narrative(ev, a, llm=llm_chat):
    view = prompt_view(ev, a)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(view, default=str)}]
    try:
        narr = parse_json(llm(msgs))
        narr["source"] = f"llm:{LLM_MODEL}"
    except LLMUnavailable as e:
        narr = template_narrative(ev, a)
        narr["source"], narr["llm_error"] = "template", str(e)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        narr = template_narrative(ev, a)
        narr["source"], narr["llm_error"] = "template", f"model reply could not be parsed ({e.__class__.__name__})"
    narr["unverified_numbers"] = verify_numbers(narr, view) if narr["source"].startswith("llm") else []
    return narr, view


# ------------------------------------------------------------------------------------------------ report
def investigate(conn, acct, use_llm=True, llm=llm_chat, save=True, t0=None, amount=None):
    ev = gather(conn, acct, t0, amount)
    if "error" in ev["trace"]:
        return {"account": acct, "error": ev["trace"]["error"]}
    a = assess(ev)
    if use_llm:
        narr, view = write_narrative(ev, a, llm)
    else:
        narr, view = template_narrative(ev, a), prompt_view(ev, a)
        narr.update(source="template", unverified_numbers=[])
    narr["caveats"] = a["rule_caveats"] + narr.get("caveats", [])
    rep = {"account": acct, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "risk_level": a["risk_level"], "recommended_action": a["recommended_action"],
           "freeze_candidates": a["freeze_candidates"], "review_accounts": a["review_accounts"],
           "scores": view["scores"], "rule_findings": a["reasons"], "narrative": narr,
           "similar_cases": ev["similar"].get("neighbours", [])[:3], "evidence": view,
           "prompt_chars": len(json.dumps(view, default=str))}
    if save:
        conn.execute("""CREATE TABLE IF NOT EXISTS investigations (id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(),
                        acct_id TEXT NOT NULL, risk_level TEXT NOT NULL, report JSONB NOT NULL)""")
        rep["investigation_id"] = conn.execute("INSERT INTO investigations (acct_id, risk_level, report) VALUES (%s,%s,%s) RETURNING id",
                                               (acct, a["risk_level"], json.dumps(rep, default=str))).fetchone()[0]
    return rep


def render(r):
    if "error" in r:
        return f"{r['account']}: {r['error']}"
    n = r["narrative"]
    L = [f"INVESTIGATION REPORT  -  account {r['account']}", f"RISK LEVEL: {r['risk_level']}   (decided by fixed rules)",
         f"scores: ML {r['scores']['ml_score']} (threshold {r['scores']['ml_threshold']}), anomaly percentile {r['scores']['anomaly_pct']}", "",
         "SUMMARY", "  " + n["summary"], "", "KEY SUSPICIOUS PATTERNS"]
    L += [f"  - {k['pattern']}: {k['evidence']}" for k in n["key_patterns"]]
    L += ["", "RULE FINDINGS (the numbers behind the risk level)"] + [f"  - {x}" for x in r["rule_findings"]]
    L += ["", "SIMILAR HISTORICAL CASES"] + [f"  - [{c['kind']}] {c['typology']} (similarity {c['similarity']}, {c['case_id'] or c['source']}): {c['summary']}"
                                              for c in r["similar_cases"]]
    L += ["", "CAVEATS"] + [f"  - {c}" for c in n["caveats"]]
    L += ["", "RECOMMENDED ACTION", "  " + r["recommended_action"]]
    if r["freeze_candidates"]:
        L.append("  Freeze candidates (for human approval): " + ", ".join(r["freeze_candidates"]))
    if r["review_accounts"]:
        L.append("  Also review: " + ", ".join(r["review_accounts"]))
    L.append("")
    L.append(f"narrative source: {n['source']}" + (f"  [{n['llm_error']}]" if n.get("llm_error") else "")
             + (f"  | numbers not found in evidence: {n['unverified_numbers']}" if n.get("unverified_numbers") else ""))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acct")
    ap.add_argument("--demo", nargs="?", const="mule", choices=["mule", "decoy"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    with psycopg.connect(DSN, autocommit=True) as conn:
        acct = a.acct
        if a.demo:
            acct, typ = conn.execute("""SELECT s.primary_acct, s.typology FROM scenarios s, splits p
                WHERE s.kind = %s AND p.name = 'test' AND s.t0 >= p.start_ts AND s.t0 < p.end_ts
                ORDER BY random() LIMIT 1""", (a.demo,)).fetchone()
            print(f"[demo] picked {acct} (ground truth: {typ})\n")
        if not acct:
            ap.error("give --acct or --demo")
        rep = investigate(conn, acct, use_llm=not a.no_llm)
        print(json.dumps(rep, indent=2, default=str) if a.json else render(rep))


if __name__ == "__main__":
    main()
