-- Per-account behavioural features for ONE time window [start, end).
-- Rendered by db/build_features.py, which fills :name, %(s)s (start), %(e)s (end),
-- %(large)s (min inbound amount to test for pass-through) and %(horizon)s (minutes).
-- Only transactions inside the window are used, so train/val/test never see each other's future.
-- Ground truth (labels, is_laundering, is_injected, case_id) is deliberately NOT used here.
DROP TABLE IF EXISTS features_{name};
CREATE TABLE features_{name} AS
WITH w AS MATERIALIZED (
    SELECT src, dst, ts, amount FROM transactions WHERE ts >= %(s)s AND ts < %(e)s
),
inb AS (
    SELECT dst AS acct_id, count(*) n_in, count(DISTINCT src) uniq_senders,
           sum(amount) total_in, avg(amount) avg_in, max(amount) max_in,
           min(ts) first_in, max(ts) last_in
    FROM w GROUP BY dst
),
outb AS (
    SELECT src AS acct_id, count(*) n_out, count(DISTINCT dst) uniq_recipients,
           sum(amount) total_out, avg(amount) avg_out, max(amount) max_out,
           min(ts) first_out, max(ts) last_out
    FROM w GROUP BY src
),
burst AS (   -- busiest 10-minute bucket (in + out)
    SELECT acct_id, max(c) max_txn_10m FROM (
        SELECT acct_id, count(*) c
        FROM (SELECT src AS acct_id, ts FROM w UNION ALL SELECT dst, ts FROM w) x
        GROUP BY acct_id, date_bin('10 minutes', ts, TIMESTAMP '2022-01-01')
    ) y GROUP BY acct_id
),
pass AS (    -- for each large inbound: how much left the account within the horizon, and how fast
    SELECT i.dst AS acct_id,
           count(*)                                                              AS n_large_in,
           count(*) FILTER (WHERE f.fwd >= 0.8 * i.amount)                        AS n_pass,
           count(*) FILTER (WHERE f.fwd >= 0.8 * i.amount AND f.dwell <= 120)     AS n_fast_pass,
           min(f.dwell) FILTER (WHERE f.fwd >= 0.5 * i.amount)                    AS min_dwell_s,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY f.dwell)
               FILTER (WHERE f.fwd >= 0.5 * i.amount)                             AS med_dwell_s,
           max(least(f.fwd / i.amount, 1.0))                                      AS max_fwd_frac
    FROM w i
    CROSS JOIN LATERAL (
        SELECT coalesce(sum(o.amount), 0) AS fwd,
               extract(epoch FROM min(o.ts) - i.ts) AS dwell
        FROM transactions o
        WHERE o.src = i.dst AND o.ts > i.ts AND o.ts < %(e)s
          AND o.ts <= i.ts + make_interval(mins => %(horizon)s)
    ) f
    WHERE i.amount >= %(large)s
    GROUP BY i.dst
),
base AS (
    SELECT coalesce(i.acct_id, o.acct_id) AS acct_id,
           coalesce(n_in,0) n_in, coalesce(n_out,0) n_out,
           coalesce(uniq_senders,0) uniq_senders, coalesce(uniq_recipients,0) uniq_recipients,
           coalesce(total_in,0) total_in, coalesce(total_out,0) total_out,
           coalesce(avg_in,0) avg_in, coalesce(avg_out,0) avg_out,
           coalesce(max_in,0) max_in, coalesce(max_out,0) max_out,
           least(first_in, first_out) first_ts, greatest(last_in, last_out) last_ts
    FROM inb i FULL JOIN outb o ON i.acct_id = o.acct_id
)
SELECT b.acct_id,
       b.n_in, b.n_out, b.uniq_senders, b.uniq_recipients,
       b.total_in, b.total_out, b.avg_in, b.avg_out, b.max_in, b.max_out,
       least(b.total_out / nullif(b.total_in, 0), 10)               AS passthrough_ratio,
       b.uniq_senders::float / nullif(b.n_in, 0)                    AS sender_diversity,
       b.uniq_recipients::float / nullif(b.n_out, 0)                AS recipient_diversity,
       b.uniq_recipients::float / nullif(b.uniq_senders, 0)         AS fan_ratio,
       abs(b.total_in - b.total_out) / nullif(greatest(b.total_in, b.total_out), 0) AS net_imbalance,
       extract(epoch FROM b.last_ts - b.first_ts)                   AS active_span_s,
       coalesce(bu.max_txn_10m, 0)                                  AS max_txn_10m,
       coalesce(p.n_large_in, 0)                                    AS n_large_in,
       coalesce(p.n_pass, 0)                                        AS n_pass,
       coalesce(p.n_fast_pass, 0)                                   AS n_fast_pass,
       p.min_dwell_s, p.med_dwell_s,
       coalesce(p.max_fwd_frac, 0)                                  AS max_fwd_frac,
       extract(epoch FROM (%(e)s::timestamp - a.opened_at)) / 86400 AS account_age_days,
       a.kyc_level, a.account_type
FROM base b
JOIN accounts a ON a.acct_id = b.acct_id
LEFT JOIN burst bu ON bu.acct_id = b.acct_id
LEFT JOIN pass  p  ON p.acct_id  = b.acct_id;
ALTER TABLE features_{name} ADD PRIMARY KEY (acct_id);

-- Ground truth for THIS window (evaluation + training target only, never a feature).
-- y = 1 only if the account is a mule AND took part in a laundering transaction inside the window.
DROP TABLE IF EXISTS labels_{name};
CREATE TABLE labels_{name} AS
SELECT f.acct_id,
       (coalesce(l.is_mule, false) AND lw.acct_id IS NOT NULL)::int AS y,
       l.label, l.typology, l.source, l.role
FROM features_{name} f
LEFT JOIN labels l ON l.acct_id = f.acct_id
LEFT JOIN (
    SELECT src AS acct_id FROM transactions WHERE ts >= %(s)s AND ts < %(e)s AND is_laundering = 1
    UNION
    SELECT dst FROM transactions WHERE ts >= %(s)s AND ts < %(e)s AND is_laundering = 1
) lw ON lw.acct_id = f.acct_id;
ALTER TABLE labels_{name} ADD PRIMARY KEY (acct_id);
ANALYZE features_{name};
ANALYZE labels_{name};
