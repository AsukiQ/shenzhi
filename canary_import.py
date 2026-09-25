import json, zipfile, glob, os
from neo4j import GraphDatabase

LIMIT = int(os.environ.get('CANARY_LIMIT','1000'))
archives = [p for p in glob.glob('/input/*.zip') if not p.endswith('/images.zip')]
records=[]
for zp in sorted(archives):
    with zipfile.ZipFile(zp) as z:
        for name in sorted(z.namelist()):
            if name.endswith('.json'):
                records.append((zp,name))
                if len(records)>=LIMIT: break
    if len(records)>=LIMIT: break

def ident(v):
    return ''.join(c if c.isalnum() or c=='_' else '_' for c in str(v))

def neo_props(p):
    out={}
    for k,v in (p or {}).items():
        if isinstance(v,(dict,list)) and any(isinstance(x,dict) for x in (v if isinstance(v,list) else [v])):
            out[str(k)] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        elif isinstance(v,(str,int,float,bool)) or v is None or isinstance(v,list):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out

driver=GraphDatabase.driver('bolt://127.0.0.1:7687',auth=('neo4j','paperddl_neo4j_2026'))
nodes=rels=0
with driver.session() as s:
  for zp,name in records:
    with zipfile.ZipFile(zp) as z: data=json.loads(z.read(name))
    graph=data.get('graph',{})
    ns={str(n.get('id')):n for n in graph.get('nodes',[]) if n.get('id') and n.get('label')}
    for n in ns.values():
      label=ident(n['label']); props=neo_props(n.get('properties'))
      s.run(f"MERGE (n:`{label}` {{id:$id}}) ON CREATE SET n += $props",id=str(n['id']),props=props).consume()
      nodes+=1
    for e in graph.get('edges',[]):
      a,b=str(e.get('source')),str(e.get('target')); typ=ident(e.get('type') or e.get('label') or 'RELATED_TO')
      if a not in ns or b not in ns: continue
      la,lb=ident(ns[a]['label']),ident(ns[b]['label'])
      s.run(f"MATCH (a:`{la}` {{id:$a}}),(b:`{lb}` {{id:$b}}) MERGE (a)-[r:`{typ}`]->(b) ON CREATE SET r += $props",a=a,b=b,props=neo_props(e.get('properties'))).consume()
      rels+=1
driver.close()
print(json.dumps({'files':len(records),'nodes_seen':nodes,'relationships_seen':rels}))
