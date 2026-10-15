"""Shared paper storage fields and conservative record classification."""
import json

RECORD_TYPES = {'canonical', 'external_full', 'reference_stub'}
SEARCH_SCOPES = {'canonical', 'external_full', 'all'}
PLACEHOLDER_ABSTRACTS = {
    'no summary was provided.', 'no abstract available', 'no abstract available.',
    'no abstract provided', 'no abstract provided.', 'n/a', 'none', 'null',
}
BUSINESS_COLUMNS = {
    'external': 'INTEGER NOT NULL DEFAULT 0',
    'record_type': "TEXT NOT NULL DEFAULT 'canonical'",
    'business_id': 'INTEGER',
    'arxiv_id': "TEXT NOT NULL DEFAULT ''",
    'source': "TEXT NOT NULL DEFAULT ''",
    'primary_subject': "TEXT NOT NULL DEFAULT ''",
    'track': "TEXT NOT NULL DEFAULT ''",
    'pdf_url': "TEXT NOT NULL DEFAULT ''",
    'entry_type': "TEXT NOT NULL DEFAULT ''",
    'editor_json': "TEXT NOT NULL DEFAULT '[]'",
    'booktitle': "TEXT NOT NULL DEFAULT ''",
    'pages': "TEXT NOT NULL DEFAULT ''",
    'publisher': "TEXT NOT NULL DEFAULT ''",
    'doi': "TEXT NOT NULL DEFAULT ''",
    'biburl': "TEXT NOT NULL DEFAULT ''",
    'bibsource': "TEXT NOT NULL DEFAULT ''",
    'source_file': "TEXT NOT NULL DEFAULT ''",
    'image_id': 'TEXT',
}
BUSINESS_FIELDS = tuple(k for k in BUSINESS_COLUMNS if k not in {'external', 'record_type', 'image_id'})


def classify_paper(props, *, is_main=False):
    # Main-paper status comes from the owning graph, never a metadata quality
    # heuristic. Missing external flags do not establish main-paper ownership.
    if is_main or props.get('external') is False:
        return 'canonical'
    abstract = str(props.get('abstract') or '').strip()
    if (str(props.get('title') or '').strip() and abstract
            and abstract.casefold() not in PLACEHOLDER_ABSTRACTS):
        return 'external_full'
    return 'reference_stub'


def ensure_business_schema(conn):
    columns = {row[1] for row in conn.execute('PRAGMA table_info(papers)')}
    for name, definition in BUSINESS_COLUMNS.items():
        if name not in columns:
            conn.execute(f'ALTER TABLE papers ADD COLUMN {name} {definition}')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_papers_business_page
        ON papers(record_type, year DESC, conference, title, row_id)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_papers_business_filters
        ON papers(record_type, conference, year DESC, track, title, row_id)''')


def business_values(doc):
    result = {name: doc.get(name) for name in BUSINESS_FIELDS}
    for name in BUSINESS_FIELDS:
        if name != 'business_id':
            result[name] = str(result[name] or '')
    if 'editor' in doc:
        result['editor_json'] = json.dumps(doc['editor'], ensure_ascii=False)
    result['editor_json'] = result.get('editor_json') or '[]'
    if 'type' in doc:
        result['entry_type'] = str(doc['type'] or '')
    subjects = doc.get('subjects') or []
    if not result['primary_subject'] and subjects:
        result['primary_subject'] = subjects[0]
    if not result['track']:
        result['track'] = result['primary_subject'].split(' - ', 1)[-1].strip() or 'Unknown'
    return result
