import csv, json, urllib.request, base64
from pathlib import Path

ROOT=Path('/root/jiansuo/demo_aaai2026'); out=ROOT/'graph'; out.mkdir(exist_ok=True)
ids=json.loads((ROOT/'scope.json').read_text())['paper_ids']
url='http://127.0.0.1:17474/db/neo4j/tx/commit'; auth=base64.b64encode(b'neo4j:paperddl_neo4j_2026').decode()
def run(stmt, params):
 req=urllib.request.Request(url,json.dumps({'statements':[{'statement':stmt,'parameters':params}]}).encode(),{'Content-Type':'application/json','Authorization':'Basic '+auth})
 d=json.loads(urllib.request.urlopen(req,timeout=60).read());
 if d.get('errors'): raise RuntimeError(d['errors'])
 r=d['results'][0]; return [x['row'] for x in r['data']]
papers={}; authors={}; topics={}; confs={}; institutions={}; edges=[]
for i in range(0,len(ids),500):
 b=ids[i:i+500]
 for pid,props in run('MATCH (p:Paper) WHERE p.paper_id IN $ids RETURN p.paper_id, properties(p)',{'ids':b}): papers[pid]=props
 for pid,aid,ap in run('MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author) WHERE p.paper_id IN $ids RETURN p.paper_id,a.author_id,properties(a)',{'ids':b}): authors[aid]=ap; edges.append((pid,aid,'AUTHORED_BY'))
 for pid,tid,tp in run('MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic) WHERE p.paper_id IN $ids RETURN p.paper_id,t.topic_id,properties(t)',{'ids':b}): topics[tid]=tp; edges.append((pid,tid,'HAS_TOPIC'))
 for pid,cid,cp in run('MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference) WHERE p.paper_id IN $ids RETURN p.paper_id,c.conference_id,properties(c)',{'ids':b}): confs[cid]=cp; edges.append((pid,cid,'PUBLISHED_IN'))
 print(i+len(b), flush=True)
for i in range(0,len(authors),500):
 b=list(authors)[i:i+500]
 for aid,iid,ip in run('MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution) WHERE a.author_id IN $ids RETURN a.author_id,i.institution_id,properties(i)',{'ids':b}):
  institutions[iid]=ip; edges.append((aid,iid,'AFFILIATED_WITH'))
def write_nodes(name, rows):
 with (out/name).open('w',newline='',encoding='utf8') as f:
  w=csv.writer(f); w.writerow(['node_id:ID','properties_json:string',':LABEL'])
  for k,p in rows.items(): w.writerow([k,json.dumps(p,ensure_ascii=False),'Node'])
write_nodes('papers.csv',papers); write_nodes('authors.csv',authors); write_nodes('topics.csv',topics); write_nodes('conferences.csv',confs)
write_nodes('institutions.csv',institutions)
with (out/'relationships.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.writer(f); w.writerow([':START_ID',':END_ID',':TYPE'])
 for a,b,t in edges:w.writerow([a,b,t])
(out/'manifest.json').write_text(json.dumps({'papers':len(papers),'authors':len(authors),'topics':len(topics),'conferences':len(confs),'institutions':len(institutions),'relationships':len(edges)},indent=2))
print(json.dumps({'papers':len(papers),'authors':len(authors),'topics':len(topics),'conferences':len(confs),'institutions':len(institutions),'relationships':len(edges)}))
