"""Fill missing main-paper authors from Neo4j; never rebuild the FTS index."""
import json
import sqlite3
import requests

ROOT = '/root/jiansuo/retrieval_backend/'
cfg = dict(line.strip().split('=', 1) for line in open(ROOT + 'deployment/paperddl.env.bm25') if '=' in line and not line.startswith('#'))
response = requests.post(cfg['NEO4J_HTTP_URI'] + '/db/neo4j/query/v2',
    auth=(cfg['NEO4J_USER'], cfg['NEO4J_PASSWORD']), timeout=120,
    json={'statement': 'MATCH (p:Paper) WHERE p.external=false OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) RETURN p.paper_id, collect(DISTINCT a.name)'})
response.raise_for_status()
data = response.json()
if data.get('errors'):
    raise RuntimeError(data['errors'])
conn = sqlite3.connect(ROOT + 'papers_fts.db', timeout=60)
conn.execute('PRAGMA synchronous=FULL')
if conn.execute('PRAGMA journal_mode').fetchone()[0] in ('off', 'memory'):
    raise RuntimeError('Unsafe journal mode')
before = dict(conn.execute('SELECT paper_id,authors_json FROM papers'))
changed = 0
with conn:
    for pid, names in data['data']['values']:
        if pid not in before or json.loads(before[pid] or '[]'):
            continue
        # Graph edges do not establish publication author order.
        names = sorted({s.strip() for s in names if isinstance(s, str) and s.strip()}, key=str.casefold)
        if not names:
            continue
        conn.execute('UPDATE papers SET authors_json=? WHERE paper_id=?', (json.dumps(names, ensure_ascii=False), pid))
        conn.execute('DELETE FROM paper_authors WHERE paper_id=?', (pid,))
        conn.executemany('INSERT OR IGNORE INTO paper_authors VALUES (?,?,?,?)',
                         [(pid, name, name.casefold(), i) for i, name in enumerate(names)])
        changed += 1
after = dict(conn.execute('SELECT paper_id,authors_json FROM papers'))
assert all(after[pid] == value for pid, value in before.items() if json.loads(value or '[]'))
missing = sum(not json.loads(after.get(pid, '[]')) for pid, _ in data['data']['values'])
print(json.dumps({'filled': changed, 'main_remaining_empty': missing, 'existing_authors_preserved': True}), flush=True)
conn.close()
