"""Product-facing metadata tables that live beside the paper index.

The tables deliberately contain only the curated Topic and Scholar overlay.
Paper records, FTS rows, and graph identities remain in their existing stores.
"""

PRODUCT_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_catalog (
    id TEXT PRIMARY KEY,
    image_path TEXT NOT NULL DEFAULT '/images/topics/default.png',
    name_zh TEXT NOT NULL DEFAULT '',
    name_en TEXT NOT NULL DEFAULT '',
    description_zh TEXT NOT NULL DEFAULT '',
    description_en TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    subject_query TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    sort_order INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_topic_catalog_enabled_category
    ON topic_catalog(enabled, category, sort_order, id);

CREATE TABLE IF NOT EXISTS topic_aliases (
    topic_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    alias_key TEXT NOT NULL,
    PRIMARY KEY (topic_id, alias_key),
    FOREIGN KEY (topic_id) REFERENCES topic_catalog(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_topic_aliases_alias
    ON topic_aliases(alias_key, topic_id);

CREATE TABLE IF NOT EXISTS scholar_profiles (
    scholar_id TEXT PRIMARY KEY,
    name_zh TEXT,
    name_en TEXT,
    avatar_path TEXT,
    bio_zh TEXT,
    bio_en TEXT,
    citation_count INTEGER,
    citation_source TEXT,
    citation_coverage TEXT,
    citation_as_of TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS scholar_affiliations (
    scholar_id TEXT NOT NULL,
    institution TEXT NOT NULL,
    position TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (scholar_id, institution, position),
    FOREIGN KEY (scholar_id) REFERENCES scholar_profiles(scholar_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_scholar_affiliations_scholar
    ON scholar_affiliations(scholar_id, institution);

CREATE TABLE IF NOT EXISTS scholar_citations (
    scholar_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (scholar_id, year),
    FOREIGN KEY (scholar_id) REFERENCES scholar_profiles(scholar_id) ON DELETE CASCADE
);
"""


def ensure_product_schema(conn):
    """Create the product overlay tables without touching paper data."""
    conn.executescript(PRODUCT_SCHEMA)

