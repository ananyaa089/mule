-- Time-respecting fund tracing. Safe to re-run (CREATE OR REPLACE).
--
-- trace_forward : where did the money GO?     Starts at the account's outbound txns after p_t0, then
--                 follows each receiving account's outbound txns, each hop within p_horizon of the previous.
-- trace_backward: where did the money COME FROM? Starts at inbound txns into the account just before p_t0,
--                 then follows the SENDER's own inbound txns, each hop within p_lookback before the next.
-- p_min_frac drops noise: a hop must carry at least this fraction of the seed amount.
-- Forward hops also cannot carry more than the previous hop received (money cannot grow), and no account is
-- revisited on a path (no cycles). Both functions stop after p_limit rows.

CREATE OR REPLACE FUNCTION trace_forward(
    p_acct text, p_t0 timestamp, p_seed double precision,
    p_horizon interval DEFAULT interval '30 minutes', p_max_depth int DEFAULT 3,
    p_min_frac double precision DEFAULT 0.05, p_limit int DEFAULT 300)
RETURNS TABLE (depth int, txn_id bigint, ts timestamp, src text, dst text, amount double precision, parent_txn bigint)
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE fwd AS (
        SELECT 1 AS depth, t.txn_id, t.ts, t.src, t.dst, t.amount, NULL::bigint AS parent_txn,
               ARRAY[t.src, t.dst] AS path
        FROM transactions t
        WHERE t.src = p_acct AND t.ts > p_t0 AND t.ts <= p_t0 + p_horizon
          AND t.amount >= p_min_frac * p_seed
      UNION ALL
        SELECT f.depth + 1, t.txn_id, t.ts, t.src, t.dst, t.amount, f.txn_id, f.path || t.dst
        FROM fwd f
        JOIN transactions t ON t.src = f.dst AND t.ts > f.ts AND t.ts <= f.ts + p_horizon
        WHERE f.depth < p_max_depth
          AND t.amount >= p_min_frac * p_seed
          AND t.amount <= f.amount * 1.001
          AND NOT (t.dst = ANY (f.path))
    )
    SELECT depth, txn_id, ts, src, dst, amount, parent_txn FROM fwd ORDER BY depth, ts LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION trace_backward(
    p_acct text, p_t0 timestamp, p_seed double precision,
    p_lookback interval DEFAULT interval '24 hours', p_max_depth int DEFAULT 2,
    p_min_frac double precision DEFAULT 0.05, p_limit int DEFAULT 300)
RETURNS TABLE (depth int, txn_id bigint, ts timestamp, src text, dst text, amount double precision, child_txn bigint)
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE bwd AS (
        SELECT 1 AS depth, t.txn_id, t.ts, t.src, t.dst, t.amount, NULL::bigint AS child_txn,
               ARRAY[t.dst, t.src] AS path
        FROM transactions t
        WHERE t.dst = p_acct AND t.ts <= p_t0 + interval '1 second' AND t.ts >= p_t0 - p_lookback
          AND t.amount >= p_min_frac * p_seed
      UNION ALL
        SELECT b.depth + 1, t.txn_id, t.ts, t.src, t.dst, t.amount, b.txn_id, b.path || t.src
        FROM bwd b
        JOIN transactions t ON t.dst = b.src AND t.ts < b.ts AND t.ts >= b.ts - p_lookback
        WHERE b.depth < p_max_depth
          AND t.amount >= p_min_frac * p_seed
          AND NOT (t.src = ANY (b.path))
    )
    SELECT depth, txn_id, ts, src, dst, amount, child_txn FROM bwd ORDER BY depth, ts DESC LIMIT p_limit;
$$;
