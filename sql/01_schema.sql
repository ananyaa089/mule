-- MuleGuard schema. Safe to re-run: it drops and recreates everything.
DROP TABLE IF EXISTS audit_log, scenarios, patterns, labels, transactions, accounts CASCADE;

CREATE TABLE accounts (
    acct_id       TEXT PRIMARY KEY,
    bank          TEXT NOT NULL,
    opened_at     TIMESTAMP NOT NULL,
    kyc_level     TEXT NOT NULL CHECK (kyc_level IN ('full','min','none')),
    account_type  TEXT NOT NULL CHECK (account_type IN ('personal','business')),
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','flagged','frozen'))
);

-- Foreign keys and indexes are added by 02_indexes.sql AFTER the bulk load (much faster).
CREATE TABLE transactions (
    txn_id        BIGINT PRIMARY KEY,
    ts            TIMESTAMP(3) NOT NULL,
    src           TEXT NOT NULL,
    dst           TEXT NOT NULL,
    amount        DOUBLE PRECISION NOT NULL CHECK (amount > 0),
    currency      TEXT NOT NULL DEFAULT 'INR',
    channel       TEXT NOT NULL,
    channel_orig  TEXT,
    is_laundering SMALLINT NOT NULL,   -- IBM / injected ground truth. NEVER a model feature.
    is_injected   BOOLEAN NOT NULL,    -- leak: evaluation/debugging only
    case_id       TEXT                 -- leak: evaluation/debugging only
);

CREATE TABLE labels (                  -- ground truth. NEVER a model feature.
    acct_id   TEXT PRIMARY KEY,
    label     TEXT NOT NULL CHECK (label IN ('mule','decoy')),
    role      TEXT,
    typology  TEXT,
    case_id   TEXT,
    source    TEXT NOT NULL CHECK (source IN ('ibm','injected')),
    is_mule   BOOLEAN NOT NULL
);

CREATE TABLE scenarios (
    case_id        TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    typology       TEXT NOT NULL,
    primary_acct   TEXT NOT NULL,
    t0             TIMESTAMP(3) NOT NULL,
    amount         NUMERIC(20,2),
    n_txns         INT,
    dwell_profile  TEXT
);

CREATE TABLE patterns (
    case_id      TEXT PRIMARY KEY,
    typology     TEXT NOT NULL,
    n_txns       INT,
    n_accounts   INT,
    start_ts     TIMESTAMP,
    end_ts       TIMESTAMP,
    total_amount NUMERIC(24,2)
);

CREATE TABLE audit_log (               -- every freeze / flag / review decision lands here
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    acct_id     TEXT NOT NULL,
    action      TEXT NOT NULL,         -- flag | freeze | unfreeze | dismiss
    actor       TEXT NOT NULL,         -- analyst name or 'system'
    risk_level  TEXT,
    reason      TEXT,
    report      JSONB                  -- the full investigation report at decision time
);
