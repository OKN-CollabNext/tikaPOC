-- File: init.sql

-- Table for Person
CREATE TABLE persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    affiliation TEXT,
    profile_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for Organization (CollegeOrUniversity)
CREATE TABLE institutions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    institution_type TEXT,  -- e.g. 'CollegeOrUniversity', 'Company', etc.
    url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create database
CREATE DATABASE openalex_topics;

-- Connect to database and create extension
\c openalex_topics
-- CREATE EXTENSION IF NOT EXISTS vector;

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
    -- embedding FLOAT8[],  -- Temporarily disabled
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Junction table for many-to-many relationship
CREATE TABLE topic_keywords (
    topic_id TEXT REFERENCES topics(id),
    keyword_id BIGINT REFERENCES keywords(id),
    PRIMARY KEY (topic_id, keyword_id)
);

-- -- Create an index for vector similarity search
-- CREATE INDEX ON keywords USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);  -- Number of lists can be adjusted based on your data size

-------------------------------------------------
-- NEW: Grants
-------------------------------------------------
CREATE TABLE grants (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    investigators TEXT[],  -- or a separate table if you prefer
    start_date DATE,
    end_date DATE,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_grants (
    topic_id TEXT REFERENCES topics(id),
    grant_id TEXT REFERENCES grants(id),
    PRIMARY KEY (topic_id, grant_id)
);

CREATE INDEX ON grants USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-------------------------------------------------
-- NEW: Patents
-------------------------------------------------
CREATE TABLE patents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    inventor TEXT[],
    publication_date DATE,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_patents (
    topic_id TEXT REFERENCES topics(id),
    patent_id TEXT REFERENCES patents(id),
    PRIMARY KEY (topic_id, patent_id)
);

CREATE INDEX ON patents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-------------------------------------------------
-- NEW: Conferences
-------------------------------------------------
CREATE TABLE conferences (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    start_date DATE,
    end_date DATE,
    description TEXT,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topic_conferences (
    topic_id TEXT REFERENCES topics(id),
    conference_id TEXT REFERENCES conferences(id),
    PRIMARY KEY (topic_id, conference_id)
);

CREATE INDEX ON conferences USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
