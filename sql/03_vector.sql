-- Needs the pgvector extension (included in the pgvector/pgvector:pg16 image).
-- Dimension 16 is a placeholder; it is set properly when the feature vector is finalised.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS cases (
    case_id    TEXT PRIMARY KEY,
    acct_id    TEXT NOT NULL,
    typology   TEXT NOT NULL,
    summary    TEXT,
    embedding  vector(16)
);
