"""Repair confirmed extraction artifacts without deleting source graph records.

Run with service environment loaded, --apply to commit, otherwise report only.
Backs up affected graph records and SQLite/FTS rows before each store is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

from funding_cleanup import clean_funding_name
from neo4j_filter import Neo4jHttpPaperFilter
from paper_search import _fts_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--db', default=str(Path(__file__).with_name('papers_fts.db')))
    args = parser.parse_args()
    graph = Neo4jHttpPaperFilter(
        http_uri=os.environ['NEO4J_HTTP_URI'], user=os.environ['NEO4J_USER'],
        password=os.environ['NEO4J_PASSWORD'], timeout_seconds=60)
    nodes = graph._run('MATCH (f:Funding) RETURN properties(f) AS props', {})
    by_name = {}
    for row in nodes:
        f = row['props']
        by_name.setdefault(f['name'].casefold().strip(), f['funding_id'])
    edits = []
    for row in nodes:
        f = row['props']
        names, reason = clean_funding_name(f['name'])
        if names == [f['name']]:
            continue
        targets = []
        for name in names:
            key = name.casefold().strip()
            fid = by_name.setdefault(key, 'funding:normalized:' + hashlib.sha256(key.encode()).hexdigest()[:24])
            targets.append({'id': fid, 'name': name})
        edits.append({'id': f['funding_id'], 'original': f, 'targets': targets, 'reason': reason})
    ids = [e['id'] for e in edits]
    edges = graph._run('''MATCH (p:Paper)-[r:FUNDED_BY]->(f:Funding)
        WHERE f.funding_id IN $ids RETURN p.paper_id AS paper_id,
        f.funding_id AS funding_id, properties(r) AS props''', {'ids': ids})
    affected = sorted({e['paper_id'] for e in edges})
    print(json.dumps({'changed_nodes': len(edits), 'quarantined_nodes': sum(not e['targets'] for e in edits),
        'split_nodes': sum(bool(e['targets']) for e in edits), 'affected_papers': len(affected)}), flush=True)
    if not args.apply or not edits:
        return
    folder = Path(__file__).with_name('data') / ('funding_repair_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    folder.mkdir(parents=True)
    (folder / 'graph_before.json').write_text(json.dumps({'edits': edits, 'edges': edges}, ensure_ascii=False))
    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA synchronous=FULL')
    if conn.execute('PRAGMA journal_mode').fetchone()[0] in ('off', 'memory'):
        raise RuntimeError('Unsafe journal mode')
    conn.execute('CREATE TEMP TABLE affected(paper_id TEXT PRIMARY KEY)')
    conn.executemany('INSERT INTO affected VALUES (?)', [(p,) for p in affected])
    papers = [dict(r) for r in conn.execute('SELECT p.* FROM papers p JOIN affected a USING(paper_id)')]
    # One scan of FTS, never one full scan per paper.
    fts = [dict(r) for r in conn.execute('SELECT f.rowid AS fts_rowid,f.* FROM papers_fts f JOIN affected a ON a.paper_id=f.paper_id')]
    relations = [dict(r) for r in conn.execute('SELECT f.* FROM paper_funding f JOIN affected a USING(paper_id)')]
    if len(papers) != len(affected) or len(fts) != len(papers) or len({r['paper_id'] for r in fts}) != len(papers):
        raise RuntimeError('Missing/duplicate SQLite or FTS records; no graph writes performed')
    (folder / 'sqlite_before.json').write_text(json.dumps({'papers': papers, 'fts': fts, 'paper_funding': relations}, ensure_ascii=False))
    conn.commit()
    graph._run('''UNWIND $edits AS edit
        MATCH (f:Funding {funding_id:edit.id})
        SET f:FundingRaw, f.cleanup_reason=edit.reason, f.cleanup_version='funding-v1'
        REMOVE f:Funding
        WITH f, edit
        UNWIND edit.targets AS target
        MERGE (t:Funding {funding_id:target.id})
        ON CREATE SET t.name=target.name
        MERGE (t)-[:NORMALIZED_FROM]->(f)
        WITH f,t
        MATCH (p:Paper)-[:FUNDED_BY]->(f)
        MERGE (p)-[:FUNDED_BY]->(t)
        RETURN count(*) AS links''', {'edits': edits})
    current = graph._run('''UNWIND $ids AS id MATCH (p:Paper {paper_id:id})
        OPTIONAL MATCH (p)-[:FUNDED_BY]->(f:Funding)
        RETURN id AS paper_id, collect(DISTINCT f.name) AS names''', {'ids': affected})
    clean = {r['paper_id']: sorted(r['names'], key=str.casefold) for r in current}
    if set(clean) != set(affected):
        raise RuntimeError('Incomplete graph response; backups at ' + str(folder))
    fts_ids = {r['paper_id']:r['fts_rowid'] for r in fts}
    with conn:
        for paper in papers:
            pid = paper['paper_id']; names = clean[pid]
            conn.execute('UPDATE papers SET funding_json=? WHERE paper_id=?', (json.dumps(names, ensure_ascii=False), pid))
            conn.execute('DELETE FROM paper_funding WHERE paper_id=?', (pid,))
            conn.executemany('INSERT OR IGNORE INTO paper_funding VALUES (?,?,?,?)', [(pid,n,n.casefold(),i) for i,n in enumerate(names)])
            text = _fts_text(' '.join(json.loads(paper['subjects_json']) + names))
            conn.execute('UPDATE papers_fts SET subjects=? WHERE rowid=?', (text, fts_ids[pid]))
        for pid,names in clean.items():
            actual = conn.execute('SELECT funding_json FROM papers WHERE paper_id=?', (pid,)).fetchone()[0]
            assert json.loads(actual) == names
    print(json.dumps({'synced_papers': len(papers), 'backup':str(folder)}), flush=True)
    conn.close()


if __name__ == '__main__':
    main()
