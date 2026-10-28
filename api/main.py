"""MuleGuard API.   Run:  uvicorn api.main:app --port 8000      (docs: http://localhost:8000/docs)

GET  /health
GET  /alerts            accounts the ML model flags (test window), highest score first
POST /investigate       run the agent on an account, store the report
GET  /trace/{account}  fund trace (where the money came from / went)
GET  /investigations    recent reports (optionally one account)   GET /investigations/{id}
POST /freeze            human-approved freeze (needs confirm=true); updates accounts.status + audit_log
POST /dismiss           analyst says "not a mule"; logged
GET  /audit             decision history
The agent only RECOMMENDS. Nothing is frozen unless /freeze is called with confirm=true.
"""
import json
import os
import sys
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import agent  # noqa: E402
import fund_trace  # noqa: E402

DSN = os.environ.get("MULEGUARD_DSN", "postgresql://muleguard:muleguard@localhost:5433/muleguard")
app = FastAPI(title="MuleGuard", version="0.1")


def db():
    return psycopg.connect(DSN, autocommit=True)


def rows(cur):
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


class InvestigateIn(BaseModel):
    account: str
    use_llm: bool = True


class FreezeIn(BaseModel):
    accounts: list[str] = Field(min_length=1)
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=3)
    confirm: bool = False
    investigation_id: int | None = None


class DismissIn(BaseModel):
    account: str
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=3)
    investigation_id: int | None = None


@app.get("/health")
def health():
    with db() as c:
        c.execute("SELECT 1")
    return {"ok": True, "llm_configured": bool(os.environ.get("GROQ_API_KEY"))}


@app.get("/alerts")
def alerts(limit: int = Query(50, ge=1, le=500), min_score: float | None = None):
    thr = fund_trace.threshold()
    lo = thr if min_score is None else min_score
    with db() as c:
        cur = c.execute("""
            SELECT m.acct_id, round(m.score::numeric, 4)::float AS ml_score, round(a.anomaly_pct::numeric, 3)::float AS anomaly_pct,
                   ac.status, f.n_fast_pass, round(f.total_in::numeric, 0)::float AS total_in,
                   EXISTS (SELECT 1 FROM investigations i WHERE i.acct_id = m.acct_id) AS investigated
            FROM ml_scores m
            LEFT JOIN anomaly_scores a ON a.split = m.split AND a.acct_id = m.acct_id
            JOIN accounts ac ON ac.acct_id = m.acct_id
            LEFT JOIN features_test f ON f.acct_id = m.acct_id
            WHERE m.split = 'test' AND m.score >= %s
            ORDER BY m.score DESC LIMIT %s""", (lo, limit))
        return {"threshold": thr, "alerts": rows(cur)}


@app.post("/investigate")
def investigate(body: InvestigateIn):
    with db() as c:
        if not c.execute("SELECT 1 FROM accounts WHERE acct_id = %s", (body.account,)).fetchone():
            raise HTTPException(404, f"unknown account {body.account}")
        rep = agent.investigate(c, body.account, use_llm=body.use_llm)
    if "error" in rep:
        raise HTTPException(422, rep["error"])
    return rep


@app.get("/trace/{account}")
def trace(account: str):
    with db() as c:
        res = fund_trace.trace_account(c, account)
    if "error" in res:
        raise HTTPException(422, res["error"])
    return res


@app.get("/investigations")
def investigations(account: str | None = None, limit: int = Query(20, ge=1, le=200)):
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS investigations (id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(),
                     acct_id TEXT NOT NULL, risk_level TEXT NOT NULL, report JSONB NOT NULL)""")
        cur = c.execute("""SELECT id, ts, acct_id, risk_level FROM investigations
                           WHERE (%s::text IS NULL OR acct_id = %s) ORDER BY id DESC LIMIT %s""", (account, account, limit))
        return rows(cur)


@app.get("/investigations/{inv_id}")
def investigation(inv_id: int):
    with db() as c:
        r = c.execute("SELECT report FROM investigations WHERE id = %s", (inv_id,)).fetchone()
    if not r:
        raise HTTPException(404, "no such investigation")
    return r[0]


@app.post("/freeze")
def freeze(body: FreezeIn):
    if not body.confirm:
        raise HTTPException(400, "confirm must be true: a human has to approve a freeze")
    done = []
    with psycopg.connect(DSN) as c:               # one transaction: status change and audit entry succeed or fail together
        report = None
        if body.investigation_id is not None:
            r = c.execute("SELECT report FROM investigations WHERE id = %s", (body.investigation_id,)).fetchone()
            if not r:
                raise HTTPException(404, "no such investigation")
            report = r[0]
        for a in dict.fromkeys(body.accounts):
            cur = c.execute("UPDATE accounts SET status = 'frozen' WHERE acct_id = %s AND status <> 'frozen' RETURNING acct_id", (a,))
            if cur.fetchone() is None:
                if not c.execute("SELECT 1 FROM accounts WHERE acct_id = %s", (a,)).fetchone():
                    raise HTTPException(404, f"unknown account {a}")
                continue                           # already frozen: no duplicate entry
            c.execute("INSERT INTO audit_log (acct_id, action, actor, risk_level, reason, report) VALUES (%s,'freeze',%s,%s,%s,%s)",
                      (a, body.actor, report and report.get("risk_level"), body.reason, json.dumps(report) if report else None))
            done.append(a)
    return {"frozen": done, "skipped_already_frozen": [a for a in dict.fromkeys(body.accounts) if a not in done]}


@app.post("/dismiss")
def dismiss(body: DismissIn):
    with psycopg.connect(DSN) as c:
        r = c.execute("SELECT report FROM investigations WHERE id = %s", (body.investigation_id,)).fetchone() if body.investigation_id else None
        if not c.execute("SELECT 1 FROM accounts WHERE acct_id = %s", (body.account,)).fetchone():
            raise HTTPException(404, f"unknown account {body.account}")
        c.execute("INSERT INTO audit_log (acct_id, action, actor, risk_level, reason, report) VALUES (%s,'dismiss',%s,%s,%s,%s)",
                  (body.account, body.actor, r and r[0].get("risk_level"), body.reason, json.dumps(r[0]) if r else None))
    return {"dismissed": body.account}


@app.get("/audit")
def audit(limit: int = Query(50, ge=1, le=500)):
    with db() as c:
        return rows(c.execute("SELECT id, ts, acct_id, action, actor, risk_level, reason FROM audit_log ORDER BY id DESC LIMIT %s", (limit,)))
