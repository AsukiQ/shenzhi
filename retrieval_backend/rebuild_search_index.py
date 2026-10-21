"""Build a validated new index from the current canonical SQLite metadata.

Never replaces the running database. Retains all API metadata and image IDs.
"""
import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from paper_search import PaperSearchIndex
from product_schema import ensure_product_schema


def copy_product_overlay(source, target):
    """Copy curated Topic/Scholar metadata while rebuilding paper tables."""
    ensure_product_schema(target)
    tables = {
        'topic_catalog': 'id,image_path,name_zh,name_en,description_zh,description_en,category,subject_query,enabled,sort_order,updated_at',
        'topic_aliases': 'topic_id,alias,alias_key',
        'scholar_profiles': 'scholar_id,name_zh,name_en,avatar_path,bio_zh,bio_en,citation_count,citation_source,citation_coverage,citation_as_of,updated_at',
        'scholar_affiliations': 'scholar_id,institution,position',
        'scholar_citations': 'scholar_id,year,count',
        'scholar_paper_links': 'scholar_id,paper_id,source',
        'scholar_details': 'scholar_id,affiliation_history_json,research_impact_json',
    }
    for table, columns in tables.items():
        try:
            rows = source.execute(f'SELECT {columns} FROM {table}').fetchall()
        except sqlite3.OperationalError as exc:
            if 'no such table' not in str(exc):
                raise
            rows = []
        if rows:
            placeholders = ','.join('?' for _ in columns.split(','))
            target.executemany(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})', [tuple(row) for row in rows])


def rebuild(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target:
        raise ValueError('Target must be a new path; never rebuild the running database in place')
    src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
    src.row_factory = sqlite3.Row
    src.execute("BEGIN")
    manifest = source.with_name('paper_images.json')
    images = json.loads(manifest.read_text()) if manifest.exists() else {}
    try:
        with tempfile.TemporaryDirectory(dir=target.parent, prefix='.index-build-') as tmp:
            documents = Path(tmp)/'documents.jsonl'
            count = 0
            with documents.open('w') as stream:
                for row in src.execute('SELECT * FROM papers ORDER BY row_id'):
                    d = dict(row)
                    for key in ('authors','keywords','subjects','funding'):
                        raw = row[key+'_json'] if key+'_json' in row.keys() else '[]'
                        d[key] = json.loads(raw or '[]')
                    entries = images.get(d['paper_id']) or []
                    if len(entries)>1: raise ValueError('Ambiguous image mapping: '+d['paper_id'])
                    d['image_id'] = row['image_id'] if 'image_id' in row.keys() else None
                    if not d['image_id'] and entries:
                        d['image_id'] = Path(entries[0]['image_path']).name
                    stream.write(json.dumps(d,ensure_ascii=False)+'\n'); count+=1
            staged = Path(tmp)/'index.db'
            index = PaperSearchIndex(staged)
            assert index.build(documents) == count
            check = sqlite3.connect(staged)
            assert check.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
            assert check.execute('SELECT count(*) FROM papers_fts').fetchone()[0] == count
            check.execute("INSERT INTO papers_fts(papers_fts) VALUES('integrity-check')")
            check.commit()
            check.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            check.close(); index.close()
            source_meta = src
            target_meta = sqlite3.connect(staged)
            try:
                with target_meta:
                    copy_product_overlay(source_meta, target_meta)
            finally:
                target_meta.close()
            # Hard link fails if target appeared concurrently; no overwrite.
            target.hardlink_to(staged)
            return count
    finally:
        src.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--target',required=True)
    args=parser.parse_args()
    print(json.dumps({'papers':rebuild(args.source,args.target),'target':args.target}))
