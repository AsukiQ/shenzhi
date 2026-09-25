"""Read-only ID coverage audit; no embedding calls or database mutations."""
import json, os, sqlite3, zipfile
from pathlib import Path
import requests
from vector_ingest import _zilliz_client

ROOT = Path(__file__).resolve().parent
for line in (ROOT/'deployment/paperddl.env.bm25').read_text().splitlines():
    if '=' in line and not line.startswith('#'):
        key, value = line.split('=', 1)
        os.environ[key] = value
conn = sqlite3.connect(f'file:{ROOT}/papers_fts.db?mode=ro', uri=True)
sql = {r[0] for r in conn.execute('SELECT paper_id FROM papers')}
r = requests.post(os.environ['NEO4J_HTTP_URI']+'/db/neo4j/query/v2',
    auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']),
    json={'statement':'MATCH (p:Paper) RETURN p.paper_id, p.external'}, timeout=120)
r.raise_for_status()
payload=r.json()
if payload.get('errors'): raise RuntimeError(payload['errors'])
graph={pid for pid, external in payload['data']['values']}
main={pid for pid, external in payload['data']['values'] if external is False}
new=set()
for archive in (ROOT.parent/'new').glob('canonical_*.zip'):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('.json'):
                papers=[n for n in json.loads(z.read(name)).get('graph',{}).get('nodes',[]) if n.get('label')=='Paper']
                if papers: new.add(papers[0]['id'])
report={'sqlite':len(sql),'neo4j':len(graph),'neo4j_main':len(main),
        'sqlite_only':len(sql-graph),'neo4j_only':len(graph-sql),
        'new_total':len(new),'new_in_sqlite':len(new&sql),'new_in_neo4j':len(new&graph)}
print(json.dumps(report),flush=True)
try:
    client=_zilliz_client(os.environ['ZILLIZ_URI'],os.environ['ZILLIZ_TOKEN'])
    iterator=client.query_iterator(collection_name=os.environ['ZILLIZ_COLLECTION'],
        filter='',output_fields=['paper_id'],batch_size=1000,timeout=60)
    dense=set(); chunks=0
    try:
        while True:
            rows=iterator.next()
            if not rows: break
            chunks+=len(rows); dense.update(row['paper_id'] for row in rows)
    finally: iterator.close()
    report.update(zilliz_chunks=chunks,zilliz_papers=len(dense),
        zilliz_only_vs_sqlite=len(dense-sql),sqlite_without_vectors=len(sql-dense),
        main_without_vectors=len(main-dense),new_in_zilliz=len(new&dense))
except Exception as exc:
    report['zilliz_audit_error']=type(exc).__name__
out=ROOT/'data/database_coverage_report.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report),flush=True)
