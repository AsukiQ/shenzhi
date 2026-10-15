#!/usr/bin/env python3
"""Export Neo4j primary papers as MySQL-shaped JSONL for reconciliation."""
import argparse,base64,json,shlex,sqlite3,urllib.request
from pathlib import Path

def settings(path):
 d={}
 for line in Path(path).read_text().splitlines():
  if line.strip() and not line.lstrip().startswith('#') and '=' in line:
   k,v=line.split('=',1);p=shlex.split(v);d[k.strip()]=p[0] if p else ''
 return d

def call(endpoint,auth,statement,params):
 req=urllib.request.Request(endpoint,json.dumps({'statements':[{'statement':statement,'parameters':params}]}).encode(),{'Content-Type':'application/json','Authorization':auth})
 with urllib.request.urlopen(req,timeout=120) as r: out=json.load(r)
 if out.get('errors'): raise RuntimeError(out['errors'])
 result=out['results'][0];return [dict(zip(result['columns'],x['row'])) for x in result['data']]

def arrays(value):
 if isinstance(value,list): return [str(x) for x in value if x not in (None,'')]
 return []

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--env',default='/home/user2/jiansuo/retrieval_backend/deployment/paperddl.env.bm25');ap.add_argument('--sqlite',default='/home/user2/jiansuo/retrieval_backend/papers_fts.db');ap.add_argument('--output',required=True);ap.add_argument('--batch',type=int,default=500);args=ap.parse_args()
 c=settings(args.env);endpoint=(c.get('NEO4J_HTTP_URI') or 'http://127.0.0.1:17474').rstrip('/')+'/db/neo4j/tx/commit';auth='Basic '+base64.b64encode(f"{c['NEO4J_USER']}:{c['NEO4J_PASSWORD']}".encode()).decode()
 ids=[r['id'] for r in call(endpoint,auth,"MATCH (p:Paper) WHERE p.external=false RETURN p.paper_id AS id ORDER BY p.paper_id",{})]
 sqlite=sqlite3.connect('file:'+str(Path(args.sqlite).resolve())+'?mode=ro',uri=True);sqlite.row_factory=sqlite3.Row;out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);stats={'neo4j_primary':len(ids),'written':0,'with_year':0,'with_conference':0,'sqlite_fill':0}
 q='''UNWIND $ids AS id MATCH (p:Paper {paper_id:id}) OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author) OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic) OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(c:Conference) OPTIONAL MATCH (p)-[:FUNDED_BY]->(f:Funding) RETURN p.paper_id AS paper_id, properties(p) AS pp, collect(DISTINCT a.name) AS authors, collect(DISTINCT t.name) AS topics, collect(DISTINCT c) AS conferences, collect(DISTINCT f.name) AS funding'''
 with out.open('w',encoding='utf8') as fh:
  for start in range(0,len(ids),args.batch):
   rows=call(endpoint,auth,q,{'ids':ids[start:start+args.batch]})
   for row in rows:
    p=row['pp'] or {};pid=row['paper_id'];srow=sqlite.execute('SELECT * FROM papers WHERE paper_id=?',(pid,)).fetchone(); s=dict(srow) if srow else {}
    authors=arrays(row['authors']) or json.loads(s.get('authors_json') or '[]');topics=arrays(row['topics']) or json.loads(s.get('subjects_json') or '[]');funding=arrays(row['funding']) or json.loads(s.get('funding_json') or '[]')
    confs=[x.get('properties',{}) for x in row['conferences'] if isinstance(x,dict)];conf=confs[0] if confs else {}
    title=str(p.get('title') or s.get('title') or '').strip(); abstract=str(p.get('abstract') or s.get('abstract') or '')
    year=p.get('year') or conf.get('year') or s.get('year');
    try: year=int(year) if year not in (None,'') else 0
    except: year=0
    venue=str(p.get('venue') or conf.get('venue') or conf.get('name') or s.get('conference') or '')
    keywords=json.loads(s.get('keywords_json') or '[]')
    rec={'paper_id':pid,'arxiv_id':str(p.get('arxiv_id') or s.get('source_id') or ''),'source':str(p.get('source') or ''),'conference':venue,'year':year,'subjects':topics,'primary_subject':topics[0] if topics else '','track':topics[0].split(' - ',1)[-1].strip() if topics else 'Unknown','title':title,'authors':authors,'abstract':abstract,'pdf_url':str(p.get('pdf_url') or p.get('url') or ''),'keywords':keywords,'type':'','editor':[],'booktitle':'','pages':'','publisher':'','doi':str(p.get('doi') or ''),'biburl':'','bibsource':'','source_file':'','funding':funding}
    fh.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n');stats['written']+=1;stats['with_year']+=bool(year);stats['with_conference']+=bool(venue);stats['sqlite_fill']+=bool(srow and (not p.get('abstract') or not row['authors']))
   print(json.dumps({'processed':min(start+args.batch,len(ids)),'total':len(ids)}),flush=True)
 sqlite.close();print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
