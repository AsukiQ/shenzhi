"""Build an independent subset index without touching production."""
import argparse
import json
import sqlite3
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--scope', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    target = Path(a.output)
    if target.exists():
        raise SystemExit('Output already exists; refusing to overwrite')
    ids = json.loads(Path(a.scope).read_text())['paper_ids']
    source = sqlite3.connect(Path(a.source).resolve().as_uri() + '?mode=ro', uri=True)
    source.row_factory = sqlite3.Row
    out = sqlite3.connect(target)
    tables = ['papers', 'paper_authors', 'paper_keywords', 'paper_subjects', 'paper_funding']
    with out:
        for table in tables + ['papers_fts']:
            sql = source.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
            out.execute(sql)
        for pos in range(0, len(ids), 400):
            batch = ids[pos:pos+400]
            marks = ','.join('?' for _ in batch)
            for table in tables:
                rows = source.execute(f'SELECT * FROM {table} WHERE paper_id IN ({marks})', batch).fetchall()
                if rows:
                    out.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in rows[0])})', [tuple(r) for r in rows])
            rows = source.execute(f'SELECT paper_id,title,abstract,keywords,subjects FROM papers_fts WHERE paper_id IN ({marks})', batch).fetchall()
            out.executemany('INSERT INTO papers_fts(paper_id,title,abstract,keywords,subjects) VALUES (?,?,?,?,?)', [tuple(r) for r in rows])
        for row in source.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"):
            out.execute(row[0])
    count = out.execute('SELECT count(*) FROM papers').fetchone()[0]
    assert count == len(set(ids)), (count, len(ids))
    assert out.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert out.execute('SELECT count(*) FROM papers_fts').fetchone()[0] == count
    out.close()
    source.close()
    print(json.dumps({'papers': count, 'bytes': target.stat().st_size, 'integrity': 'ok'}))


if __name__ == '__main__':
    main()
