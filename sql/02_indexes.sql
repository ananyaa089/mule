-- Run after loading. These indexes are what make dwell-time and trace queries fast.
ALTER TABLE transactions ADD CONSTRAINT fk_txn_src FOREIGN KEY (src) REFERENCES accounts(acct_id);
ALTER TABLE transactions ADD CONSTRAINT fk_txn_dst FOREIGN KEY (dst) REFERENCES accounts(acct_id);
ALTER TABLE labels       ADD CONSTRAINT fk_lab_acct FOREIGN KEY (acct_id) REFERENCES accounts(acct_id);

CREATE INDEX idx_txn_src_ts ON transactions (src, ts);   -- "what did this account send, in order"
CREATE INDEX idx_txn_dst_ts ON transactions (dst, ts);   -- "what did this account receive, in order"
CREATE INDEX idx_txn_ts     ON transactions USING brin (ts);  -- tiny, ts is physically sorted
CREATE INDEX idx_txn_big    ON transactions (ts) WHERE amount >= 100000;  -- alert scans
CREATE INDEX idx_acct_status ON accounts (status) WHERE status <> 'active';

ANALYZE;
