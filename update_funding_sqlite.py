import sqlite3,zipfile,json
from collections import defaultdict
from pathlib import Path
archive='/root/jiansuo/AAAI-2026.zip'; out=defaultdict(set)
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():
  if not n.endswith('.json'): continue
  d=json.loads(z.read(n)); g=d.get('graph',{}); title=None
  for x in g.get('nodes',[]):
   if x.get('label')=='Paper': title=str(x.get('properties',{}).get('title','')).strip().lower()
  if not title: continue
  for e in g.get('edges',[]):
   if e.get('type')=='FUNDED_BY':
    for x in g.get('nodes',[]):
     if x.get('id')==e.get('target') and x.get('label')=='Funding': out[title].add(x.get('properties',{}).get('name') or x.get('id'))
for dbp in ['/root/jiansuo/retrieval_backend/papers_fts.db','/root/jiansuo/demo_aaai2026/papers_fts.db']:
 c=sqlite3.connect(dbp); rows=c.execute('select paper_id,title from papers').fetchall(); updates=0
 for pid,title in rows:
  vals=sorted(out.get(str(title).strip().lower(),set()))
  if not vals: continue
  c.execute('update papers set funding_json=? where paper_id=?',(json.dumps(vals,ensure_ascii=False),pid)); c.execute('delete from paper_funding where paper_id=?',(pid,))
  for i,v in enumerate(vals): c.execute('insert into paper_funding(paper_id,value,value_key,position) values(?,?,?,?)',(pid,v,v.casefold(),i))
  updates+=1
 c.commit(); c.close(); print(dbp,updates)
