import json, zipfile, glob, os
from neo4j import GraphDatabase

LIMIT = int(os.getenv('LIMIT', '0'))
CK = os.getenv('CHECKPOINT', '/checkpoint')
CHUNK = int(os.getenv('CHUNK', '500'))
def ident(v):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in str(v))
def props(p):
    out = {}
    for k, v in (p or {}).items():
        if isinstance(v, dict) or (isinstance(v, list) and any(isinstance(x, dict) for x in v)):
            v = json.dumps(v, ensure_ascii=False, sort_keys=True)
        out[str(k)] = v if isinstance(v, (str, int, float, bool, list)) or v is None else str(v)
    return out
files = []
for zp in sorted(glob.glob('/input/*.zip')):
    if zp.endswith('/images.zip'): continue
    with zipfile.ZipFile(zp) as z:
        files += [(zp, n) for n in sorted(z.namelist()) if n.endswith('.json')]
start = int(open(CK).read()) if os.path.exists(CK) else 0
if LIMIT: files = files[:LIMIT]
d = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', 'paperddl_neo4j_2026'))
for i, (zp, nm) in enumerate(files[start:], start):
    with zipfile.ZipFile(zp) as z: g = json.loads(z.read(nm)).get('graph', {})
    ns = {str(n.get('id')): n for n in g.get('nodes', []) if n.get('id') and n.get('label')}
    with d.session() as s:
        for label in set(ident(n['label']) for n in ns.values()):
            rows = [{'id': k, 'props': props(n.get('properties'))} for k, n in ns.items() if ident(n['label']) == label]
            for off in range(0, len(rows), CHUNK):
                s.run(f"UNWIND $rows AS x MERGE (n:`{label}` {{id:x.id}}) ON CREATE SET n += x.props", rows=rows[off:off+CHUNK]).consume()
        groups = {}
        for e in g.get('edges', []):
            a, b = str(e.get('source')), str(e.get('target')); typ = ident(e.get('type') or e.get('label') or 'RELATED_TO')
            if a in ns and b in ns: groups.setdefault((ident(ns[a]['label']), typ, ident(ns[b]['label'])), []).append({'a': a, 'b': b, 'props': props(e.get('properties'))})
        for (la, typ, lb), rows in groups.items():
            for off in range(0, len(rows), CHUNK):
                s.run(f"UNWIND $rows AS x MATCH (a:`{la}` {{id:x.a}}),(b:`{lb}` {{id:x.b}}) MERGE (a)-[r:`{typ}`]->(b) ON CREATE SET r += x.props", rows=rows[off:off+CHUNK]).consume()
    open(CK, 'w').write(str(i + 1))
    if (i + 1) % 100 == 0: print('checkpoint', i + 1, flush=True)
d.close(); print('complete', len(files), flush=True)
