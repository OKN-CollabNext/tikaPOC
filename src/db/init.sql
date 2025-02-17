CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE topics (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE keywords (
    id BIGSERIAL PRIMARY KEY,
    keyword TEXT NOT NULL,
    topic_id TEXT REFERENCES topics(id),
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (keyword, topic_id)
);

CREATE INDEX ON keywords USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_keywords_trgm ON keywords USING gin (keyword gin_trgm_ops);
CREATE INDEX idx_keywords_topic_id ON keywords(topic_id);
CREATE INDEX idx_keywords_keyword ON keywords(keyword);

CREATE OR REPLACE FUNCTION rrf(dense_rank float, keyword_rank float, k float default 60.0)
RETURNS float AS $$
BEGIN
    RETURN 1.0/(k + dense_rank) + 1.0/(k + keyword_rank);
END;
$$ LANGUAGE plpgsql;

UPDATE keywords SET embedding = NULL;
