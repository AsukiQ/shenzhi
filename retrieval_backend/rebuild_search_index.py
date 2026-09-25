"""Build a validated new index from the current canonical SQLite metadata.

Never replaces the running database. Retains all API metadata and image IDs.
"""
import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from paper_search import PaperSearchIndex


def rebuild(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target:
        raise ValueError('Target must be a new path; never rebuild the running database in place')
    src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
    src.row_factory = sqlite3.Row
    manifest = source.with_name('paper_images.json')
    images = json.loads(manifest.read_text()) if manifest.exists() else {}
    try:
        with tempfile.TemporaryDirectory(dir=target.parent, prefix='.index-build-') as tmp:
            documents = Path(tmp)/'documents.jsonl'
            count = 0
            with documents.open('w') as stream:
                for row in src.execute('SELECT * FROM papers ORDER BY row_id'):
                    d = {key: row[key] for key in ('paper_id','source_id','title','abstract','conference','year')}
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
