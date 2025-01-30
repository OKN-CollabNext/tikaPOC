-- Create database
CREATE DATABASE openalex_topics;

-- Connect to database and create extension
\c openalex_topics
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables with proper relationships
CREATE TABLE topics (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE keywords (
    id BIGSERIAL PRIMARY KEY,
    keyword TEXT NOT NULL UNIQUE,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Junction table for many-to-many relationship
CREATE TABLE topic_keywords (
    topic_id TEXT REFERENCES topics(id) ON DELETE CASCADE,
    keyword_id BIGINT REFERENCES keywords(id) ON DELETE CASCADE,
    PRIMARY KEY (topic_id, keyword_id)
);

-- Table for hierarchical relationships
CREATE TABLE topic_hierarchy (
    ancestor_id TEXT REFERENCES topics(id) ON DELETE CASCADE,
    child_id TEXT REFERENCES topics(id) ON DELETE CASCADE,
    PRIMARY KEY (ancestor_id, child_id)
);

-- Create indexes for hierarchical relationships
CREATE INDEX IF NOT EXISTS idx_topic_hierarchy_ancestor ON topic_hierarchy (ancestor_id);
CREATE INDEX IF NOT EXISTS idx_topic_hierarchy_child ON topic_hierarchy (child_id);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_keywords_embedding ON keywords USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Number of lists can be adjusted based on your data size
