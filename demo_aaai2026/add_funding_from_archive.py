import csv,json,zipfile,sqlite3
from pathlib import Path
ROOT=Path('/root/jiansuo/demo_aaai2026'); G=ROOT/'graph'; scope=set(json.loads((ROOT/'scope.json').read_text())['paper_ids'])
db=sqlite3.connect(ROOT/'papers_fts.db'); title_map={str(t).strip().lower():pid for pid,t in db.execute('select paper_id,title from papers')}
funds={}; edges=set()
with zipfile.ZipFile('/root/jiansuo/AAAI-2026.zip') as z:
 for n in z.namelist():
  if not n.endswith('.json'): continue
  d=json.loads(z.read(n)); g=d.get('graph',{})
  for x in g.get('nodes',[]):
   if x.get('label')=='Funding': funds[x['id']]=x.get('properties',{})
  local={x.get('id'): title_map.get(str(x.get('properties',{}).get('title','')).strip().lower()) for x in g.get('nodes',[]) if x.get('label')=='Paper'}
  for e in g.get('edges',[]):
   if e.get('type')=='FUNDED_BY' and local.get(e.get('source')) in scope: edges.add((local[e['source']],e['target'],'FUNDED_BY'))
used={b for a,b,t in edges}; funds={k:v for k,v in funds.items() if k in used}
with (G/'fundings.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.writer(f); w.writerow(['node_id:ID','properties_json:string',':LABEL'])
 for k,p in funds.items(): w.writerow([k,json.dumps(p,ensure_ascii=False),'Funding'])
with (G/'relationships_funding.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.writer(f); w.writerow([':START_ID',':END_ID',':TYPE'])
 for x in sorted(edges): w.writerow(x)
m=json.loads((G/'manifest.json').read_text()); m.update({'fundings':len(funds),'funding_relationships':len(edges)}); (G/'manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps({'fundings':len(funds),'funding_relationships':len(edges)}))
